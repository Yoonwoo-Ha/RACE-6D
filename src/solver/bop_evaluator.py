"""Official BOP19 pose evaluation from validation postprocessor outputs."""

import json
import os
import subprocess
import sys
from pathlib import Path

import torch

from ..misc import dist_utils


class BOPPoseEvaluator:
    """Collect BOP poses across ranks and run the official toolkit on rank 0."""

    SCORE_KEYS = {
        "bop_vsd": "bop19_average_recall_vsd",
        "bop_mssd": "bop19_average_recall_mssd",
        "bop_mspd": "bop19_average_recall_mspd",
        "bop_ar": "bop19_average_recall",
    }

    def __init__(
        self,
        config,
        coco_api,
        label_to_category,
        labels_are_categories,
        output_dir,
        epoch=None,
    ):
        self.config = dict(config)
        self.coco_api = coco_api
        self.label_to_category = {
            int(label): int(category) for label, category in label_to_category.items()
        }
        self.labels_are_categories = bool(labels_are_categories)
        self.output_dir = Path(output_dir).expanduser().resolve()
        self.epoch = epoch
        self.records = {}

        self.dataset = self.config.get("dataset", "lmo")
        self.split = self.config.get("split", "test")
        self.method = self.config.get("method", "race6deval")

    def update(self, targets, results):
        for target, result in zip(targets, results):
            image_id = int(target["image_id"].item())
            image_info = self.coco_api.loadImgs(image_id)[0]
            scene_id = int(image_info["scene_id"])
            original_name = image_info.get("original_filename", image_info["file_name"])
            im_id = int(Path(original_name).stem)

            required = ("labels", "scores", "rotations", "translations")
            missing = [key for key in required if key not in result]
            if missing:
                raise KeyError(f"BOP evaluation requires postprocessor outputs: {missing}")

            labels = result["labels"].detach().cpu().tolist()
            scores = result["scores"].detach().cpu().tolist()
            rotations = result["rotations"].detach().cpu().reshape(-1, 9).tolist()
            translations = result["translations"].detach().cpu().reshape(-1, 3).tolist()

            image_records = []
            for label, score, rotation, translation in zip(
                labels, scores, rotations, translations
            ):
                label = int(label)
                if self.labels_are_categories:
                    obj_id = label
                else:
                    if label not in self.label_to_category:
                        continue
                    obj_id = self.label_to_category[label]
                image_records.append(
                    {
                        "scene_id": scene_id,
                        "im_id": im_id,
                        "obj_id": obj_id,
                        "score": float(score),
                        "R": rotation,
                        "t": translation,
                    }
                )

            self.records[image_id] = image_records

    def summarize(self):
        gathered = dist_utils.all_gather(self.records)
        payload = [None]

        if dist_utils.is_main_process():
            try:
                merged = {}
                for rank_records in gathered:
                    for image_id, records in rank_records.items():
                        merged.setdefault(image_id, records)
                payload[0] = {"metrics": self._evaluate(merged), "error": None}
            except Exception as exc:
                payload[0] = {"metrics": None, "error": str(exc)}

        if dist_utils.is_dist_available_and_initialized():
            torch.distributed.broadcast_object_list(payload, src=0)

        if payload[0]["error"] is not None:
            raise RuntimeError(f"BOP evaluation failed: {payload[0]['error']}")
        return payload[0]["metrics"]

    def _evaluate(self, records_by_image):
        project_root = Path(__file__).resolve().parents[2]
        toolkit_path = self._resolve_path(
            self.config.get("toolkit_path", "~/Downloads/bop_toolkit"), project_root
        )
        renderer_path = self._resolve_path(
            self.config.get("renderer_path", "~/Downloads/bop_renderer/build"), project_root
        )
        dataset_root = Path(
            os.path.expanduser(
                self.config.get("dataset_root", f"~/Downloads/{self.dataset}")
            )
        )
        eval_script = toolkit_path / "scripts" / "eval_bop19_pose.py"
        targets_path = dataset_root / self.config.get(
            "targets_filename", "test_targets_bop19.json"
        )
        if not eval_script.is_file():
            raise FileNotFoundError(f"BOP toolkit script not found: {eval_script}")
        if not targets_path.is_file():
            raise FileNotFoundError(f"BOP targets not found: {targets_path}")

        results_path = self.output_dir / "bop_results"
        eval_path = self.output_dir / "bop_eval_results"
        scores_path = self.output_dir / "bop_scores"
        results_path.mkdir(parents=True, exist_ok=True)
        eval_path.mkdir(parents=True, exist_ok=True)
        scores_path.mkdir(parents=True, exist_ok=True)

        result_filename = f"{self.method}_{self.dataset}-{self.split}.csv"
        result_file = results_path / result_filename
        self._write_csv(result_file, records_by_image)

        env = os.environ.copy()
        env["BOP_PATH"] = str(dataset_root.parent)
        python_path = [str(renderer_path), str(toolkit_path)]
        if env.get("PYTHONPATH"):
            python_path.append(env["PYTHONPATH"])
        env["PYTHONPATH"] = os.pathsep.join(python_path)

        command = [
            sys.executable,
            str(eval_script),
            f"--renderer_type={self.config.get('renderer_type', 'cpp')}",
            f"--result_filenames={result_filename}",
            f"--results_path={results_path}",
            f"--eval_path={eval_path}",
            f"--targets_filename={targets_path.name}",
            f"--num_workers={int(self.config.get('num_workers', 8))}",
        ]
        completed = subprocess.run(
            command,
            cwd=toolkit_path,
            env=env,
            text=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
        )
        (self.output_dir / "bop_eval_latest.log").write_text(
            completed.stdout, encoding="utf-8"
        )
        if completed.returncode != 0:
            tail = "\n".join(completed.stdout.splitlines()[-40:])
            raise RuntimeError(
                f"BOP toolkit exited with code {completed.returncode}:\n{tail}"
            )

        result_name = result_filename[:-4]
        score_file = eval_path / result_name / "scores_bop19.json"
        with score_file.open("r", encoding="utf-8") as file:
            official_scores = json.load(file)

        epoch_name = "eval" if self.epoch is None else f"epoch{int(self.epoch):04d}"
        score_snapshot = scores_path / f"{epoch_name}.json"
        score_snapshot.write_text(
            json.dumps(official_scores, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )

        metrics = {
            output_key: float(official_scores[official_key])
            for output_key, official_key in self.SCORE_KEYS.items()
        }
        print(
            "BOP19: "
            f"VSD={metrics['bop_vsd']:.4f} "
            f"MSSD={metrics['bop_mssd']:.4f} "
            f"MSPD={metrics['bop_mspd']:.4f} "
            f"AR={metrics['bop_ar']:.4f}"
        )
        return metrics

    @staticmethod
    def _resolve_path(path, project_root):
        resolved = Path(os.path.expanduser(path))
        if not resolved.is_absolute():
            resolved = project_root / resolved
        return resolved.resolve()

    @staticmethod
    def _write_csv(path, records_by_image):
        records = [
            record
            for image_records in records_by_image.values()
            for record in image_records
        ]
        records.sort(
            key=lambda item: (
                item["scene_id"],
                item["im_id"],
                -item["score"],
                item["obj_id"],
            )
        )
        with path.open("w", encoding="utf-8") as file:
            for record in records:
                rotation = " ".join(f"{value:.8f}" for value in record["R"])
                translation = " ".join(f"{value:.2f}" for value in record["t"])
                file.write(
                    f"{record['scene_id']},{record['im_id']},{record['obj_id']},"
                    f"{record['score']:.8f},{rotation},{translation},0.0\n"
                )
