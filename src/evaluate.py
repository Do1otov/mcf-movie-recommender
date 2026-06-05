import numpy as np
import torch
import pandas as pd
from src.config import CFG


@torch.no_grad()
def get_predictions(model, loader, device, model_type: str):
    model.eval()
    all_pred, all_true = [], []

    for batch in loader:
        user_idx = batch["user_idx"].to(device)
        item_idx = batch["item_idx"].to(device)
        metadata = batch["metadata"].to(device)
        true_overall = batch["overall"].numpy()

        if model_type == "baseline":
            pred = model(user_idx, item_idx, metadata).cpu().numpy()
        elif model_type in ("no_meta", "fixed_weights"):
            pred, _ = model(user_idx, item_idx, metadata)
            pred = pred.cpu().numpy()
        else:
            pred, _, _ = model(user_idx, item_idx, metadata)
            pred = pred.cpu().numpy()

        all_pred.append(pred)
        all_true.append(true_overall)

    return np.concatenate(all_pred), np.concatenate(all_true)


def ndcg_at_k(pred: np.ndarray, true: np.ndarray, k: int = 10) -> float:
    """NDCG@K для одного пользователя."""
    order = np.argsort(pred)[::-1][:k]
    gains = true[order]
    discounts = np.log2(np.arange(2, len(gains) + 2))
    dcg = np.sum(gains / discounts)

    ideal_order = np.argsort(true)[::-1][:k]
    ideal_gains = true[ideal_order]
    idcg = np.sum(ideal_gains / discounts[: len(ideal_gains)])

    return float(dcg / idcg) if idcg > 0 else 0.0


def precision_at_k(
    pred: np.ndarray, true: np.ndarray, k: int = 10, threshold: float = 3.5
) -> float:
    """Precision@K: доля релевантных (true >= threshold) в топ-K предсказаний."""
    top_k_idx = np.argsort(pred)[::-1][:k]
    relevant = (true[top_k_idx] >= threshold).sum()
    return float(relevant / k)


def evaluate_ranking(
    model, test_df: pd.DataFrame, device, model_type: str, k: int = 10
):
    """
    Считаем NDCG@K и Precision@K по каждому пользователю, усредняем.
    """
    from src.dataset import MovieDataset
    from torch.utils.data import DataLoader

    ndcg_scores, prec_scores = [], []

    for user_idx_val, group in test_df.groupby("user_idx"):
        if len(group) < k:
            continue

        ds = MovieDataset(group)
        loader = DataLoader(ds, batch_size=len(group), shuffle=False)
        pred, true = get_predictions(model, loader, device, model_type)

        ndcg_scores.append(ndcg_at_k(pred, true, k))
        prec_scores.append(precision_at_k(pred, true, k))

    return {
        f"ndcg@{k}": float(np.mean(ndcg_scores)),
        f"precision@{k}": float(np.mean(prec_scores)),
    }


def full_evaluation(model, test_loader, test_df, device, model_type: str, name: str):
    from src.train import eval_epoch

    print(f"\n{'=' * 50}")
    print(f"Оценка модели: {name}")
    print(f"{'=' * 50}")

    metrics = eval_epoch(model, test_loader, device, model_type)
    ranking = evaluate_ranking(model, test_df, device, model_type)
    metrics.update(ranking)

    for k, v in metrics.items():
        print(f"  {k}: {v:.4f}")

    return metrics
