"""Copyright(c) 2023 lyuwenyu. All Rights Reserved.
"""

import math
import torch.optim as optim
import torch.optim.lr_scheduler as lr_scheduler

from ..core import register


__all__ = ['AdamW', 'SGD', 'Adam', 'MultiStepLR', 'CosineAnnealingLR', 'OneCycleLR', 'LambdaLR', 'FlatCosineAnnealingLR']



SGD = register()(optim.SGD)
Adam = register()(optim.Adam)
AdamW = register()(optim.AdamW)


MultiStepLR = register()(lr_scheduler.MultiStepLR)
CosineAnnealingLR = register()(lr_scheduler.CosineAnnealingLR)
OneCycleLR = register()(lr_scheduler.OneCycleLR)
LambdaLR = register()(lr_scheduler.LambdaLR)


def _flat_cosine_schedule(total_iter, warmup_iter, flat_iter, current_iter, init_lr, min_lr):
    """LR schedule (iteration-based).
    Phase 1: Warmup (quadratic, 0 → init_lr)
    Phase 2: Flat (maintain init_lr)
    Phase 3: Cosine decay from the end of the flat phase to min_lr at T_max
    Phase 4: min_lr for any epoch after T_max (training longer than T_max holds the final LR)
    """
    cosine_end = total_iter
    if current_iter <= warmup_iter:
        return init_lr * (current_iter / max(float(warmup_iter), 1)) ** 2
    elif warmup_iter < current_iter <= flat_iter:
        return init_lr
    elif current_iter >= cosine_end:
        return min_lr
    else:
        cosine_total = cosine_end - flat_iter
        progress = (current_iter - flat_iter) / max(cosine_total, 1)
        cosine_decay = 0.5 * (1 + math.cos(math.pi * progress))
        return min_lr + (init_lr - min_lr) * cosine_decay


@register()
class FlatCosineAnnealingLR:
    """FlatCosineLRScheduler (iteration-based, with warmup).

    iter_per_epoch is unknown at registry creation time, so the solver must call
    init_schedule(iter_per_epoch) after the dataloader is created.

    Args:
        optimizer: optimizer instance
        lr_gamma: min_lr = base_lr * lr_gamma (default: 0.5)
        T_max: epoch at which the cosine decay reaches min_lr; epochs beyond it stay at min_lr
        warmup_iter: number of warmup iterations (default: 2000)
        flat_epochs: number of flat phase epochs
        group_flat_epochs: dict {group_idx: flat_epochs} — per-group flat phase customization
            e.g.: {0: 0, 1: 0} → groups 0 and 1 skip flat phase and go straight to cosine decay
    """
    def __init__(self, optimizer, lr_gamma=0.5, T_max=40,
                 warmup_iter=2000, flat_epochs=5,
                 group_flat_epochs=None, **kwargs):
        if kwargs.get('no_aug_epochs'):
            raise ValueError('FlatCosineAnnealingLR: no_aug_epochs was removed; use T_max - no_aug_epochs as T_max '
                             '(the cosine then ends at the same epoch and later epochs stay at min_lr)')
        self.base_lrs = [group['lr'] for group in optimizer.param_groups]
        self.min_lrs = [lr * lr_gamma for lr in self.base_lrs]
        self.T_max = T_max
        self.warmup_iter = warmup_iter
        self.flat_epochs = flat_epochs
        self.group_flat_epochs = group_flat_epochs or {}
        self.lr_funcs = None  # per-group schedule functions
        self._iter_based = True

    def init_schedule(self, iter_per_epoch):
        """Called after dataloader is created. Computes the actual schedule from iter_per_epoch."""
        from functools import partial
        total_iter = int(iter_per_epoch * self.T_max)

        self.lr_funcs = []
        for i in range(len(self.base_lrs)):
            group_flat = self.group_flat_epochs.get(i, self.flat_epochs)
            flat_iter = int(iter_per_epoch * group_flat)
            self.lr_funcs.append(partial(
                _flat_cosine_schedule, total_iter, self.warmup_iter, flat_iter))

    def step(self, current_iter, optimizer):
        """Called every iteration. Updates the optimizer LR."""
        if self.lr_funcs is None:
            return
        for i, group in enumerate(optimizer.param_groups):
            group['lr'] = self.lr_funcs[i](current_iter, self.base_lrs[i], self.min_lrs[i])

    def get_last_lr(self):
        return self.base_lrs

    def state_dict(self):
        return {'base_lrs': self.base_lrs, 'min_lrs': self.min_lrs}

    def load_state_dict(self, state_dict):
        self.base_lrs = state_dict['base_lrs']
        self.min_lrs = state_dict['min_lrs']
