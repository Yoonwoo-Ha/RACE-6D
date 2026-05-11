# src/data/dataset/_mixed_dataset.py
import torch
from torch.utils.data import Dataset
import numpy as np
import copy

from ...core import register


@register()
class MixedDomainDataset(Dataset):
    """
    여러 도메인의 dataset을 특정 비율로 샘플링
    """

    __share__ = ["num_classes"]

    def __init__(self, datasets, sampling_ratios, use_real_length=False, **kwargs):
        """
        Args:
            datasets: list of Dataset objects or configs (dicts)
            sampling_ratios: list of float, 각 dataset의 샘플링 비율
            use_real_length: if True, use only Real dataset length (not sum of all)
        """
        from ...core.workspace import GLOBAL_CONFIG

        assert len(datasets) == len(sampling_ratios)
        assert abs(sum(sampling_ratios) - 1.0) < 1e-6

        self.datasets = []
        for ds_cfg in datasets:
            if isinstance(ds_cfg, dict):
                dataset = self._create_dataset(ds_cfg, GLOBAL_CONFIG)
                self.datasets.append(dataset)
            else:
                self.datasets.append(ds_cfg)

        self.sampling_ratios = sampling_ratios
        self.use_real_length = use_real_length

        # 각 dataset의 크기
        self.dataset_sizes = [len(d) for d in self.datasets]

        # 전체 epoch 크기 설정
        if self.use_real_length and hasattr(self, '_real_length'):
            self.total_size = self._real_length
            print(f"Mixed Dataset Info (Real only mode):")
        else:
            self.total_size = sum(self.dataset_sizes)
            print(f"Mixed Dataset Info:")

        for i, (size, ratio) in enumerate(zip(self.dataset_sizes, sampling_ratios)):
            print(f"  Dataset {i}: {size} samples, ratio: {ratio:.2%}")
        print(f"  Total virtual size: {self.total_size}")

        # 초기 인덱스 생성
        self.indices = None
        self.set_epoch(0)

    def _create_dataset(self, ds_cfg, global_cfg):
        ds_cfg = copy.deepcopy(ds_cfg)

        if "type" not in ds_cfg:
            raise ValueError("Each dataset config must have 'type' key")

        ds_type = str(ds_cfg["type"])

        if ds_type not in global_cfg:
            raise ValueError(f"Dataset type '{ds_type}' is not registered")

        schema = global_cfg[ds_type]

        _cfg = {}
        if "_kwargs" in schema and schema["_kwargs"]:
            _cfg.update(copy.deepcopy(schema["_kwargs"]))
        _cfg.update(ds_cfg)
        _cfg.pop("type", None)

        for share_key in schema.get("_share", []):
            if share_key in global_cfg and share_key not in ds_cfg:
                _cfg[share_key] = global_cfg[share_key]

        for inject_key in schema.get("_inject", []):
            inject_val = _cfg.get(inject_key)
            if inject_val is not None:
                if isinstance(inject_val, dict):
                    _cfg[inject_key] = self._create_injected(inject_val, global_cfg)
                elif isinstance(inject_val, str):
                    if inject_val in global_cfg:
                        from ...core import create

                        _cfg[inject_key] = create(inject_val, global_cfg)

        module = getattr(schema["_pymodule"], ds_type)
        module_kwargs = {k: v for k, v in _cfg.items() if not k.startswith("_")}
        return module(**module_kwargs)

    def _create_injected(self, cfg, global_cfg):
        cfg = copy.deepcopy(cfg)

        if "type" not in cfg:
            from ..transforms.container import Compose

            return Compose(**cfg)

        _type = str(cfg["type"])
        if _type not in global_cfg:
            raise ValueError(f"Type '{_type}' is not registered")

        schema = global_cfg[_type]

        _cfg = {}
        if "_kwargs" in schema and schema["_kwargs"]:
            _cfg.update(copy.deepcopy(schema["_kwargs"]))
        _cfg.update(cfg)
        _cfg.pop("type", None)

        module = getattr(schema["_pymodule"], _type)
        module_kwargs = {k: v for k, v in _cfg.items() if not k.startswith("_")}
        return module(**module_kwargs)

    def _generate_indices(self, epoch):
        """Epoch마다 데이터셋 샘플링 인덱스를 미리 생성하여 골고루 뽑히도록 함"""
        rng = np.random.RandomState(epoch)

        all_indices = []

        # 각 데이터셋별 할당량 계산
        counts = [int(self.total_size * r) for r in self.sampling_ratios]
        # 반올림 오차 보정 (마지막 데이터셋에 몰아주기)
        counts[-1] = self.total_size - sum(counts[:-1])

        for i, (size, count) in enumerate(zip(self.dataset_sizes, counts)):
            if size == 0 or count == 0:
                continue

            # 1. 필요한 만큼 반복 (Full repeats)
            n_repeat = count // size
            n_remain = count % size

            indices = []

            # 전체 데이터셋을 n_repeat 번 추가 (매번 셔플)
            for _ in range(n_repeat):
                perm = rng.permutation(size)
                indices.append(perm)

            # 남은 개수만큼 추가 (셔플 후 앞부분)
            if n_remain > 0:
                perm = rng.permutation(size)
                indices.append(perm[:n_remain])

            if indices:
                indices = np.concatenate(indices)
                # (dataset_idx, sample_idx) 형태로 저장
                d_idxs = np.full(len(indices), i, dtype=np.int32)
                combined = np.stack((d_idxs, indices), axis=1)
                all_indices.append(combined)

        if all_indices:
            self.indices = np.concatenate(all_indices, axis=0)
            rng.shuffle(self.indices)
        else:
            self.indices = np.zeros((0, 2), dtype=np.int32)

    def __len__(self):
        return self.total_size

    def __getitem__(self, idx):
        # 미리 생성된 매핑 테이블 사용
        if self.indices is None:
            self._generate_indices(0)

        indices = self.indices
        assert indices is not None
        dataset_idx, sample_idx = indices[idx]
        # numpy.int64 -> int 변환 (torchvision 호환성)
        return self.datasets[dataset_idx][int(sample_idx)]

    @property
    def coco(self):
        """CocoDetection 호환성을 위해"""
        if self.datasets:
            return self.datasets[0].coco
        return None

    def set_epoch(self, epoch):
        """epoch 설정 전파 및 인덱스 재생성"""
        # 하위 데이터셋에 전파
        for dataset in self.datasets:
            if hasattr(dataset, "set_epoch"):
                dataset.set_epoch(epoch)

        # 현재 epoch에 맞는 인덱스 매핑 생성
        self._generate_indices(epoch)

