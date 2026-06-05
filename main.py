import torch
import pandas as pd
from src.config import CFG
from src.dataset import load_and_split, get_loaders
from src.models import BaselineNCF, MCF_NoMeta, MCF_FixedWeights, MCF_Full
from src.train import train_model
from src.evaluate import full_evaluation

if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Устройство: {device}")

    train_df, val_df, test_df, n_users, n_items, item_ids = load_and_split(
        CFG.data_path
    )
    train_loader, val_loader, test_loader = get_loaders(train_df, val_df, test_df)

    models_cfg = [
        ("baseline", BaselineNCF(n_users, n_items), "baseline"),
        ("no_meta", MCF_NoMeta(n_users, n_items), "mcf_no_meta"),
        ("fixed_weights", MCF_FixedWeights(n_users, n_items), "mcf_fixed_weights"),
        ("full", MCF_Full(n_users, n_items), "mcf_full"),
    ]

    all_metrics = {}

    for model_type, model, name in models_cfg:
        trained = train_model(model, train_loader, val_loader, device, model_type, name)
        metrics = full_evaluation(
            trained, test_loader, test_df, device, model_type, name
        )
        all_metrics[name] = metrics

    print("\n" + "=" * 70)
    print("ИТОГОВОЕ СРАВНЕНИЕ МОДЕЛЕЙ")
    print("=" * 70)
    results = pd.DataFrame(all_metrics).T
    print(results.to_string())
    results.to_csv("results.csv")
