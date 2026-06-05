import pandas as pd
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split
from src.config import CFG


def load_and_split(path: str):
    df = pd.read_csv(path)

    user_ids = {uid: idx for idx, uid in enumerate(df["user_id"].unique())}
    item_ids = {iid: idx for idx, iid in enumerate(df["movie_id"].unique())}

    df["user_idx"] = df["user_id"].map(user_ids)
    df["item_idx"] = df["movie_id"].map(item_ids)

    n_users = len(user_ids)
    n_items = len(item_ids)

    train_val, test = train_test_split(
        df, test_size=CFG.test_size, random_state=CFG.random_seed
    )
    val_ratio = CFG.val_size / (1 - CFG.test_size)
    train, val = train_test_split(
        train_val, test_size=val_ratio, random_state=CFG.random_seed
    )

    print(f"Train: {len(train)} | Val: {len(val)} | Test: {len(test)}")
    print(f"Users: {n_users} | Items: {n_items}")

    return train, val, test, n_users, n_items, item_ids


class MovieDataset(Dataset):
    def __init__(self, df: pd.DataFrame):
        self.user_idx = torch.tensor(df["user_idx"].values, dtype=torch.long)
        self.item_idx = torch.tensor(df["item_idx"].values, dtype=torch.long)

        genre_cols = [f"genre_{i}" for i in range(CFG.num_genres)]
        meta = df[["year_norm"] + genre_cols].values.astype(np.float32)
        self.metadata = torch.tensor(meta, dtype=torch.float32)

        self.overall = torch.tensor(df["overall"].values, dtype=torch.float32)
        criteria_vals = df[CFG.criteria].values.astype(np.float32)
        self.criteria = torch.tensor(criteria_vals, dtype=torch.float32)

    def __len__(self):
        return len(self.user_idx)

    def __getitem__(self, idx):
        return {
            "user_idx": self.user_idx[idx],
            "item_idx": self.item_idx[idx],
            "metadata": self.metadata[idx],
            "overall": self.overall[idx],
            "criteria": self.criteria[idx],
        }


def get_loaders(train, val, test):
    return (
        DataLoader(
            MovieDataset(train),
            batch_size=CFG.batch_size,
            shuffle=True,
            pin_memory=True,
        ),
        DataLoader(
            MovieDataset(val),
            batch_size=CFG.batch_size,
            shuffle=False,
            pin_memory=True,
        ),
        DataLoader(
            MovieDataset(test),
            batch_size=CFG.batch_size,
            shuffle=False,
            pin_memory=True,
        ),
    )
