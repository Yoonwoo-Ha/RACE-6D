"""Copyright(c) 2023 lyuwenyu. All Rights Reserved."""

import math
import re
import time
import json
import datetime
import torch

from ..misc import dist_utils
from ._solver import BaseSolver
from .pose_engine import train_one_epoch, evaluate


class PoseSolver(BaseSolver):
    _BOP_CKPT_PATTERN = re.compile(
        r"^ckp(?P<epoch>\d{4})_top(?P<rank>\d+)_(?P<score>\d+)\.pth$"
    )
    _MAP_CKPT_PATTERN = re.compile(
        r"^ckp(?P<epoch>\d{4})_map_top(?P<rank>\d+)_(?P<score>\d+)\.pth$"
    )

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    def fit(
        self,
    ):
        print("Start training")
        self.train()
        args = self.cfg

        n_parameters = sum(
            [p.numel() for p in self.model.parameters() if p.requires_grad]
        )
        print(f"number of trainable parameters: {n_parameters}")

        start_time = time.time()
        start_epcoch = self.last_epoch + 1

        for epoch in range(start_epcoch, args.epoches):
            self.train_dataloader.set_epoch(epoch)
            if dist_utils.is_dist_available_and_initialized():
                self.train_dataloader.sampler.set_epoch(epoch)

            self.criterion.update_epoch(epoch)

            # If iter-based scheduler, pass to engine; otherwise use legacy warmup
            _iter_based = getattr(self.lr_scheduler, '_iter_based', False)
            train_stats = train_one_epoch(
                self.model,
                self.criterion,
                self.train_dataloader,
                self.optimizer,
                self.device,
                epoch,
                max_norm=args.clip_max_norm,
                print_freq=args.print_freq,
                ema=self.ema,
                scaler=self.scaler,
                lr_warmup_scheduler=None if _iter_based else self.lr_warmup_scheduler,
                lr_scheduler=self.lr_scheduler if _iter_based else None,
                writer=self.writer,
            )

            if not _iter_based:
                if self.lr_warmup_scheduler is None or self.lr_warmup_scheduler.finished():
                    self.lr_scheduler.step()

            self.last_epoch += 1

            if self.output_dir:
                # Always preserve the latest train state before external BOP evaluation.
                dist_utils.save_on_master(self.state_dict(), self.output_dir / "last.pth")

                # Configs without BOP evaluation retain the periodic checkpoints;
                # with it, the top-k checkpoints by BOP AR and by COCO mAP are kept instead.
                bop_eval_config = args.yaml_cfg.get("bop_eval") or {}
                if (
                    not bop_eval_config.get("enabled", False)
                    and (epoch + 1) % args.checkpoint_freq == 0
                ):
                    dist_utils.save_on_master(
                        self.state_dict(),
                        self.output_dir / f"checkpoint{epoch:04}.pth",
                    )

            module = self.ema.module if self.ema else self.model
            test_stats, coco_evaluator = evaluate(
                module,
                self.criterion,
                self.postprocessor,
                self.val_dataloader,
                self.evaluator,
                device=self.device,
                bop_eval_config=args.yaml_cfg.get("bop_eval"),
                output_dir=self.output_dir,
                epoch=epoch,
            )

            if self.output_dir and bop_eval_config.get("enabled", False):
                if "bop_ar" not in test_stats:
                    raise KeyError("BOP evaluation is enabled but bop_ar is missing")
                self._update_bop_topk(
                    epoch=epoch,
                    bop_ar=test_stats["bop_ar"],
                    top_k=int(bop_eval_config.get("checkpoint_topk", 5)),
                )
                coco_bbox_stats = test_stats.get("coco_eval_bbox")
                if coco_bbox_stats:
                    self._update_map_topk(
                        epoch=epoch,
                        coco_map=coco_bbox_stats[0],
                        top_k=int(
                            bop_eval_config.get(
                                "map_checkpoint_topk",
                                bop_eval_config.get("checkpoint_topk", 5),
                            )
                        ),
                    )

            log_stats = {
                **{f"train_{k}": v for k, v in train_stats.items()},
                **{f"test_{k}": v for k, v in test_stats.items()},
                "epoch": epoch,
                "n_parameters": n_parameters,
            }

            if self.output_dir and dist_utils.is_main_process():
                with (self.output_dir / "log.txt").open("a") as f:
                    f.write(json.dumps(log_stats) + "\n")

                # for evaluation logs
                if coco_evaluator is not None:
                    (self.output_dir / "eval").mkdir(exist_ok=True)
                    if "bbox" in coco_evaluator.coco_eval:
                        filenames = ["latest.pth"]
                        if epoch % 50 == 0:
                            filenames.append(f"{epoch:03}.pth")
                        for name in filenames:
                            torch.save(
                                coco_evaluator.coco_eval["bbox"].eval,
                                self.output_dir / "eval" / name,
                            )

        total_time = time.time() - start_time
        total_time_str = str(datetime.timedelta(seconds=int(total_time)))
        print("Training time {}".format(total_time_str))

    def val(
        self,
    ):
        self.eval()

        module = self.ema.module if self.ema else self.model
        test_stats, coco_evaluator = evaluate(
            module,
            self.criterion,
            self.postprocessor,
            self.val_dataloader,
            self.evaluator,
            self.device,
            bop_eval_config=self.cfg.yaml_cfg.get("bop_eval"),
            output_dir=self.output_dir,
            epoch=self.last_epoch if self.last_epoch >= 0 else None,
        )

        log_stats = {**{f"test_{k}": v for k, v in test_stats.items()}}

        if self.output_dir and dist_utils.is_main_process():
            with (self.output_dir / "log_test.txt").open("a") as f:
                f.write(json.dumps(log_stats) + "\n")

        if self.output_dir:
            dist_utils.save_on_master(
                coco_evaluator.coco_eval["bbox"].eval, self.output_dir / "eval.pth"
            )

        return

    def _load_bop_topk(self):
        manifest_path = self.output_dir / "bop_top5.json"
        entries = []
        if manifest_path.is_file():
            try:
                manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
                for entry in manifest.get("checkpoints", []):
                    filename = str(entry["file"])
                    if (self.output_dir / filename).is_file():
                        entries.append(
                            {
                                "epoch": int(entry["epoch"]),
                                "ar": float(entry["ar"]),
                                "file": filename,
                            }
                        )
            except (KeyError, TypeError, ValueError, json.JSONDecodeError):
                entries = []

        if entries:
            return entries

        for path in self.output_dir.glob("ckp*_top*_*.pth"):
            match = self._BOP_CKPT_PATTERN.match(path.name)
            if match:
                entries.append(
                    {
                        "epoch": int(match.group("epoch")),
                        "ar": int(match.group("score")) / 10000.0,
                        "file": path.name,
                    }
                )
        return entries

    def _update_bop_topk(self, epoch, bop_ar, top_k=5):
        if not dist_utils.is_main_process():
            return
        if top_k <= 0:
            raise ValueError(f"checkpoint_topk must be positive, got {top_k}")

        epoch = int(epoch)
        bop_ar = float(bop_ar)
        if not math.isfinite(bop_ar):
            raise ValueError(f"bop_ar must be finite, got {bop_ar}")
        existing = self._load_bop_topk()
        candidates = [entry for entry in existing if entry["epoch"] != epoch]
        candidates.append({"epoch": epoch, "ar": bop_ar, "file": None})
        ranked = sorted(candidates, key=lambda entry: (-entry["ar"], entry["epoch"]))[
            :top_k
        ]
        retained_epochs = {entry["epoch"] for entry in ranked}
        current_is_retained = epoch in retained_epochs

        current_temp = self.output_dir / f".ckp{epoch:04d}_bop_candidate.pth.tmp"
        if current_is_retained:
            torch.save(self.state_dict(), current_temp)

        staged = {}
        for entry in existing:
            path = self.output_dir / entry["file"]
            if entry["epoch"] in retained_epochs and entry["epoch"] != epoch:
                temp_path = self.output_dir / f".ckp{entry['epoch']:04d}_bop_top.pth.tmp"
                path.replace(temp_path)
                staged[entry["epoch"]] = temp_path
            else:
                path.unlink(missing_ok=True)

        manifest_entries = []
        for rank, entry in enumerate(ranked, start=1):
            score = int(round(entry["ar"] * 10000.0))
            filename = f"ckp{entry['epoch']:04d}_top{rank}_{score:04d}.pth"
            destination = self.output_dir / filename
            if entry["epoch"] == epoch:
                current_temp.replace(destination)
            else:
                staged[entry["epoch"]].replace(destination)
            manifest_entries.append(
                {
                    "rank": rank,
                    "epoch": entry["epoch"],
                    "ar": entry["ar"],
                    "file": filename,
                }
            )

        manifest = {
            "metric": "bop19_average_recall",
            "top_k": top_k,
            "checkpoints": manifest_entries,
        }
        manifest_path = self.output_dir / "bop_top5.json"
        manifest_temp = self.output_dir / ".bop_top5.json.tmp"
        manifest_temp.write_text(
            json.dumps(manifest, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        manifest_temp.replace(manifest_path)
        summary = ", ".join(
            f"top{entry['rank']}=ep{entry['epoch']:04d}/{entry['ar']:.4f}"
            for entry in manifest_entries
        )
        print(f"BOP checkpoint top-{top_k}: {summary}")

    def _load_map_topk(self):
        manifest_path = self.output_dir / "map_top5.json"
        entries = []
        if manifest_path.is_file():
            try:
                manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
                for entry in manifest.get("checkpoints", []):
                    filename = str(entry["file"])
                    if (self.output_dir / filename).is_file():
                        entries.append(
                            {
                                "epoch": int(entry["epoch"]),
                                "map": float(entry["map"]),
                                "file": filename,
                            }
                        )
            except (KeyError, TypeError, ValueError, json.JSONDecodeError):
                entries = []

        if entries:
            return entries

        for path in self.output_dir.glob("ckp*_map_top*_*.pth"):
            match = self._MAP_CKPT_PATTERN.match(path.name)
            if match:
                entries.append(
                    {
                        "epoch": int(match.group("epoch")),
                        "map": int(match.group("score")) / 10000.0,
                        "file": path.name,
                    }
                )
        return entries

    def _update_map_topk(self, epoch, coco_map, top_k=5):
        if not dist_utils.is_main_process():
            return
        if top_k <= 0:
            raise ValueError(f"map_checkpoint_topk must be positive, got {top_k}")

        epoch = int(epoch)
        coco_map = float(coco_map)
        if not math.isfinite(coco_map):
            raise ValueError(f"coco_map must be finite, got {coco_map}")
        existing = self._load_map_topk()
        candidates = [entry for entry in existing if entry["epoch"] != epoch]
        candidates.append({"epoch": epoch, "map": coco_map, "file": None})
        ranked = sorted(candidates, key=lambda entry: (-entry["map"], entry["epoch"]))[
            :top_k
        ]
        retained_epochs = {entry["epoch"] for entry in ranked}
        current_is_retained = epoch in retained_epochs

        current_temp = self.output_dir / f".ckp{epoch:04d}_map_candidate.pth.tmp"
        if current_is_retained:
            torch.save(self.state_dict(), current_temp)

        staged = {}
        for entry in existing:
            path = self.output_dir / entry["file"]
            if entry["epoch"] in retained_epochs and entry["epoch"] != epoch:
                temp_path = self.output_dir / f".ckp{entry['epoch']:04d}_map_top.pth.tmp"
                path.replace(temp_path)
                staged[entry["epoch"]] = temp_path
            else:
                path.unlink(missing_ok=True)

        manifest_entries = []
        for rank, entry in enumerate(ranked, start=1):
            score = int(round(entry["map"] * 10000.0))
            filename = (
                f"ckp{entry['epoch']:04d}_map_top{rank}_{score:04d}.pth"
            )
            destination = self.output_dir / filename
            if entry["epoch"] == epoch:
                current_temp.replace(destination)
            else:
                staged[entry["epoch"]].replace(destination)
            manifest_entries.append(
                {
                    "rank": rank,
                    "epoch": entry["epoch"],
                    "map": entry["map"],
                    "file": filename,
                }
            )

        manifest = {
            "metric": "coco_bbox_ap_50_95",
            "top_k": top_k,
            "checkpoints": manifest_entries,
        }
        manifest_path = self.output_dir / "map_top5.json"
        manifest_temp = self.output_dir / ".map_top5.json.tmp"
        manifest_temp.write_text(
            json.dumps(manifest, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        manifest_temp.replace(manifest_path)
        summary = ", ".join(
            f"top{entry['rank']}=ep{entry['epoch']:04d}/{entry['map']:.4f}"
            for entry in manifest_entries
        )
        print(f"COCO mAP checkpoint top-{top_k}: {summary}")
