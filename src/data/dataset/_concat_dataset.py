# src/data/dataset/_concat_dataset.py
import copy

from ...core import register


@register()
class ConcatDataset:
    """
    여러 dataset을 단순히 일렬로 이어붙이는 Dataset.
    """

    __share__ = ["num_classes"]

    def __init__(self, datasets, **kwargs):
        from ...core.workspace import GLOBAL_CONFIG

        self.datasets = []
        for ds_cfg in datasets:
            if isinstance(ds_cfg, dict):
                dataset = self._create_dataset(ds_cfg, GLOBAL_CONFIG)
                self.datasets.append(dataset)
            else:
                self.datasets.append(ds_cfg)

        self.dataset_sizes = [len(d) for d in self.datasets]
        self.total_size = sum(self.dataset_sizes)

        # cumulative offsets for index mapping
        self._offsets = []
        offset = 0
        for size in self.dataset_sizes:
            self._offsets.append(offset)
            offset += size

        print(f"ConcatDataset Info:")
        for i, size in enumerate(self.dataset_sizes):
            print(f"  Dataset {i}: {size} samples")
        print(f"  Total size: {self.total_size}")

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

    def __len__(self):
        return self.total_size

    def __getitem__(self, idx):
        for i in range(len(self.datasets) - 1, -1, -1):
            if idx >= self._offsets[i]:
                return self.datasets[i][idx - self._offsets[i]]
        raise IndexError(f"Index {idx} out of range for ConcatDataset of size {self.total_size}")

    @property
    def coco(self):
        if self.datasets:
            return self.datasets[0].coco
        return None

    def set_epoch(self, epoch):
        for dataset in self.datasets:
            if hasattr(dataset, "set_epoch"):
                dataset.set_epoch(epoch)
