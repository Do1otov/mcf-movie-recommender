import os
import torch
import torch.nn as nn
import numpy as np
from src.config import CFG


def compute_loss(
    pred_overall,
    pred_criteria,
    true_overall,
    true_criteria,
) -> torch.Tensor:
    """
    Суммарный loss: MSE по критериям + λ * MSE по overall.
    """
    loss_criteria = nn.functional.mse_loss(pred_criteria, true_criteria)
    loss_overall = nn.functional.mse_loss(pred_overall, true_overall)
    return loss_criteria + CFG.lambda_overall * loss_overall


def train_epoch(model, loader, optimizer, device, model_type: str) -> float:
    model.train()
    total_loss = 0.0

    for batch in loader:
        user_idx = batch["user_idx"].to(device)
        item_idx = batch["item_idx"].to(device)
        metadata = batch["metadata"].to(device)
        overall = batch["overall"].to(device)
        criteria = batch["criteria"].to(device)

        optimizer.zero_grad()

        if model_type == "baseline":
            pred_overall = model(user_idx, item_idx, metadata)
            loss = nn.functional.mse_loss(pred_overall, overall)

        elif model_type in ("no_meta", "fixed_weights"):
            pred_overall, pred_criteria = model(user_idx, item_idx, metadata)
            loss = compute_loss(pred_overall, pred_criteria, overall, criteria)

        else:
            pred_overall, pred_criteria, _ = model(user_idx, item_idx, metadata)
            loss = compute_loss(pred_overall, pred_criteria, overall, criteria)

        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()
        total_loss += loss.item()

    return total_loss / len(loader)


@torch.no_grad()
def eval_epoch(model, loader, device, model_type: str) -> dict:
    model.eval()
    all_pred_overall = []
    all_true_overall = []
    all_pred_criteria = []
    all_true_criteria = []

    for batch in loader:
        user_idx = batch["user_idx"].to(device)
        item_idx = batch["item_idx"].to(device)
        metadata = batch["metadata"].to(device)
        overall = batch["overall"]
        criteria = batch["criteria"]

        if model_type == "baseline":
            pred_overall = model(user_idx, item_idx, metadata).cpu()
            all_pred_overall.append(pred_overall)
            all_true_overall.append(overall)

        elif model_type in ("no_meta", "fixed_weights"):
            pred_overall, pred_criteria = model(user_idx, item_idx, metadata)
            all_pred_overall.append(pred_overall.cpu())
            all_true_overall.append(overall)
            all_pred_criteria.append(pred_criteria.cpu())
            all_true_criteria.append(criteria)

        else:
            pred_overall, pred_criteria, _ = model(user_idx, item_idx, metadata)
            all_pred_overall.append(pred_overall.cpu())
            all_true_overall.append(overall)
            all_pred_criteria.append(pred_criteria.cpu())
            all_true_criteria.append(criteria)

    pred_o = torch.cat(all_pred_overall).numpy()
    true_o = torch.cat(all_true_overall).numpy()
    rmse_overall = float(np.sqrt(np.mean((pred_o - true_o) ** 2)))

    metrics = {"rmse_overall": rmse_overall}

    if all_pred_criteria:
        pred_c = torch.cat(all_pred_criteria).numpy()
        true_c = torch.cat(all_true_criteria).numpy()
        for i, name in enumerate(CFG.criteria):
            metrics[f"rmse_{name}"] = float(
                np.sqrt(np.mean((pred_c[:, i] - true_c[:, i]) ** 2))
            )

    return metrics


def train_model(model, train_loader, val_loader, device, model_type: str, name: str):
    model = model.to(device)
    optimizer = torch.optim.Adam(
        model.parameters(), lr=CFG.lr, weight_decay=CFG.weight_decay
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=25, eta_min=1e-5
    )

    os.makedirs(CFG.checkpoint_dir, exist_ok=True)
    best_rmse = float("inf")
    patience_counter = 0

    print(f"\n{'=' * 50}")
    print(f"Обучение модели: {name}")
    print(f"{'=' * 50}")

    for epoch in range(1, CFG.epochs + 1):
        train_loss = train_epoch(model, train_loader, optimizer, device, model_type)
        val_metrics = eval_epoch(model, val_loader, device, model_type)
        val_rmse = val_metrics["rmse_overall"]

        scheduler.step()

        print(
            f"Epoch {epoch:3d} | train_loss={train_loss:.4f} | val_rmse_overall={val_rmse:.4f}",
            end="",
        )
        if "rmse_plot" in val_metrics:
            print(
                f" | plot={val_metrics['rmse_plot']:.4f} visual={val_metrics['rmse_visual']:.4f} acting={val_metrics['rmse_acting']:.4f} emotion={val_metrics['rmse_emotion']:.4f}",
                end="",
            )
        print()

        if val_rmse < best_rmse:
            best_rmse = val_rmse
            patience_counter = 0
            torch.save(model.state_dict(), f"{CFG.checkpoint_dir}/{name}_best.pt")
        else:
            patience_counter += 1
            if patience_counter >= CFG.patience:
                print(f"Early stopping на эпохе {epoch}")
                break

    model.load_state_dict(torch.load(f"{CFG.checkpoint_dir}/{name}_best.pt"))
    print(f"Лучший val RMSE: {best_rmse:.4f}")
    return model
