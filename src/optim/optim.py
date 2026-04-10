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


def _flat_cosine_schedule(total_iter, warmup_iter, flat_iter, no_aug_iter, current_iter, init_lr, min_lr):
    """4-phase LR schedule (iter 기반).
    Phase 1: Warmup (quadratic, 0 → init_lr)
    Phase 2: Flat (init_lr 유지)
    Phase 3: Cosine decay (init_lr → min_lr, 완전 완료)
    Phase 4: No-aug (min_lr 고정)

    총 학습 = T_max epochs, no_aug는 T_max 내 마지막 구간.
    cosine은 flat 끝 → (T_max - no_aug) 구간에서 min_lr까지 완전 감소.
    """
    cosine_end = total_iter - no_aug_iter
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
    """FlatCosineLRScheduler (iter 기반, warmup 포함).

    Registry에서 생성 시 iter_per_epoch를 모르므로, solver에서
    dataloader 생성 후 init_schedule(iter_per_epoch)를 호출해야 함.

    Args:
        optimizer: optimizer instance
        lr_gamma: min_lr = base_lr * lr_gamma (기본: 0.5)
        T_max: 전체 학습 epoch 수
        warmup_iter: warmup iteration 수 (기본: 2000)
        flat_epochs: flat phase epoch 수
        no_aug_epochs: no-aug phase epoch 수
        group_flat_epochs: dict {group_idx: flat_epochs} — group별 flat phase 커스텀
            예: {0: 0, 1: 0} → group 0,1은 flat 없이 바로 cosine 감소
    """
    def __init__(self, optimizer, lr_gamma=0.5, T_max=40,
                 warmup_iter=2000, flat_epochs=5, no_aug_epochs=8,
                 group_flat_epochs=None, **kwargs):
        self.base_lrs = [group['lr'] for group in optimizer.param_groups]
        self.min_lrs = [lr * lr_gamma for lr in self.base_lrs]
        self.T_max = T_max
        self.warmup_iter = warmup_iter
        self.flat_epochs = flat_epochs
        self.no_aug_epochs = no_aug_epochs
        self.group_flat_epochs = group_flat_epochs or {}
        self.lr_funcs = None  # per-group schedule functions
        self._iter_based = True

    def init_schedule(self, iter_per_epoch):
        """Dataloader 생성 후 호출. iter_per_epoch로 실제 schedule 계산."""
        from functools import partial
        total_iter = int(iter_per_epoch * self.T_max)
        no_aug_iter = int(iter_per_epoch * self.no_aug_epochs)

        self.lr_funcs = []
        for i in range(len(self.base_lrs)):
            group_flat = self.group_flat_epochs.get(i, self.flat_epochs)
            flat_iter = int(iter_per_epoch * group_flat)
            self.lr_funcs.append(partial(
                _flat_cosine_schedule, total_iter, self.warmup_iter, flat_iter, no_aug_iter))

    def step(self, current_iter, optimizer):
        """매 iteration 호출. optimizer의 LR을 업데이트."""
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
