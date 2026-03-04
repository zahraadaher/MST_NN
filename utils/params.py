import json
from dataclasses import dataclass
from pathlib import Path


@dataclass
class Params:
    data_path: Path
    out_dir: Path

    n_epochs: int
    batch_size_train: int
    batch_size_val: int
    lr: float
    weight_decay: float
    patience: int
    factor: float
    early_stop_patience: int
    save_interval: int

    in_channels: int
    out_channels: int
    base_features: int
    depth: int
    use_resblock: bool

    num_workers_train: int
    num_workers_val: int
    seed: int

    @classmethod
    def from_json(cls, json_path: str):
        with open(json_path, "r") as f:
            data = json.load(f)

        data["data_path"] = Path(data["data_path"])
        data["out_dir"] = Path(data["out_dir"])

        return cls(**data)

    def save(self, path: Path):
        """Save config used for this run."""
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)

        serializable = self.__dict__.copy()
        serializable["data_path"] = str(serializable["data_path"])
        serializable["out_dir"] = str(serializable["out_dir"])

        with open(path, "w") as f:
            json.dump(serializable, f, indent=4)