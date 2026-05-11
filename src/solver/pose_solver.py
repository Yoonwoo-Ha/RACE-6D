"""Copyright(c) 2023 lyuwenyu. All Rights Reserved."""

import time
import json
import datetime
import torch

from ..misc import dist_utils
from ._solver import BaseSolver
from .pose_engine import train_one_epoch, evaluate


class PoseSolver(BaseSolver):
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
                checkpoint_paths = [self.output_dir / "last.pth"]
                # extra checkpoint before LR drop and every 100 epochs
                if (epoch + 1) % args.checkpoint_freq == 0:
                    checkpoint_paths.append(
                        self.output_dir / f"checkpoint{epoch:04}.pth"
                    )
                for checkpoint_path in checkpoint_paths:
                    dist_utils.save_on_master(self.state_dict(), checkpoint_path)

            module = self.ema.module if self.ema else self.model
            test_stats, coco_evaluator = evaluate(
                module,
                self.criterion,
                self.postprocessor,
                self.val_dataloader,
                self.evaluator,
                device=self.device,
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
