from dataclasses import dataclass


@dataclass
class Config:
    data_path: str = "data/dataset_model.csv"
    checkpoint_dir: str = "checkpoints"

    test_size: float = 0.1
    val_size: float = 0.1
    random_seed: int = 42

    embed_dim: int = 64
    metadata_dim: int = 32
    hidden_dims: list = None
    dropout: float = 0.4

    batch_size: int = 1024
    lr: float = 1e-3
    weight_decay: float = 1e-4
    epochs: int = 60
    patience: int = 12
    lambda_overall: float = 1.0

    num_genres: int = 20
    criteria: list = None

    def __post_init__(self):
        if self.hidden_dims is None:
            self.hidden_dims = [128, 64]
        if self.criteria is None:
            self.criteria = ["plot", "visual", "acting", "emotion"]


CFG = Config()
