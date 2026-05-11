# src/data/dataset/_mixed_dataset.py
import torch
from torch.utils.data import Dataset
import numpy as np
import copy

from ...core import register


@register()
class MixedDomainDataset(Dataset):
    """
    Sample multiple domain datasets at specified ratios.
    """

    __share__ = ["num_classes"]

    def __init__(self, datasets, sampling_ratios, use_real_length=False, **kwargs):
        """
        Args:
            datasets: list of Dataset objects or configs (dicts)
            sampling_ratios: list of float, sampling ratio for each dataset
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

        # Size of each dataset
        self.dataset_sizes = [len(d) for d in self.datasets]

        # Set the total epoch size
        if self.use_real_length and hasattr(self, '_real_length'):
            self.total_size = self._real_length
            print(f"Mixed Dataset Info (Real only mode):")
        else:
            self.total_size = sum(self.dataset_sizes)
            print(f"Mixed Dataset Info:")

        for i, (size, ratio) in enumerate(zip(self.dataset_sizes, sampling_ratios)):
            print(f"  Dataset {i}: {size} samples, ratio: {ratio:.2%}")
        print(f"  Total virtual size: {self.total_size}")

        # Initialize indices
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
        """Pre-generate dataset sampling indices per epoch so samples are drawn evenly."""
        rng = np.random.RandomState(epoch)

        all_indices = []

        # Compute per-dataset allocation
        counts = [int(self.total_size * r) for r in self.sampling_ratios]
        # Correct rounding error (assign remainder to the last dataset)
        counts[-1] = self.total_size - sum(counts[:-1])

        for i, (size, count) in enumerate(zip(self.dataset_sizes, counts)):
            if size == 0 or count == 0:
                continue

            # 1. Repeat as needed (full repeats)
            n_repeat = count // size
            n_remain = count % size

            indices = []

            # Add the full dataset n_repeat times (shuffle each time)
            for _ in range(n_repeat):
                perm = rng.permutation(size)
                indices.append(perm)

            # Add the remaining count (front of a fresh shuffle)
            if n_remain > 0:
                perm = rng.permutation(size)
                indices.append(perm[:n_remain])

            if indices:
                indices = np.concatenate(indices)
                # Store as (dataset_idx, sample_idx) pairs
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
        # Use the pre-generated mapping table
        if self.indices is None:
            self._generate_indices(0)

        indices = self.indices
        assert indices is not None
        dataset_idx, sample_idx = indices[idx]
        # numpy.int64 -> int conversion (torchvision compatibility)
        return self.datasets[dataset_idx][int(sample_idx)]

    @property
    def coco(self):
        """For CocoDetection compatibility."""
        if self.datasets:
            return self.datasets[0].coco
        return None

    def set_epoch(self, epoch):
        """Propagate the epoch setting and regenerate indices."""
        # Propagate to child datasets
        for dataset in self.datasets:
            if hasattr(dataset, "set_epoch"):
                dataset.set_epoch(epoch)

        # Generate the index mapping for the current epoch
        self._generate_indices(epoch)

