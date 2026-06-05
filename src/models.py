import torch
import torch.nn as nn
from src.config import CFG


def build_mlp(
    in_dim: int, hidden_dims: list, out_dim: int, dropout: float
) -> nn.Sequential:
    layers = []
    prev = in_dim
    for h in hidden_dims:
        layers += [nn.Linear(prev, h), nn.ReLU(), nn.Dropout(dropout)]
        prev = h
    layers.append(nn.Linear(prev, out_dim))
    return nn.Sequential(*layers)


class MetadataEncoder(nn.Module):
    """Кодирует year_norm + multi-hot жанры в вектор фиксированной размерности."""

    def __init__(self, meta_in: int, meta_out: int, dropout: float):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(meta_in, 64),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(64, meta_out),
            nn.ReLU(),
        )

    def forward(self, metadata: torch.Tensor) -> torch.Tensor:
        return self.net(metadata)


class BaselineNCF(nn.Module):
    """
    Однокритериальная модель: предсказывает только overall.
    UserEmb + ItemEmb + MetadataEnc → MLP → scalar.
    """

    def __init__(self, n_users: int, n_items: int):
        super().__init__()
        d = CFG.embed_dim
        m = CFG.metadata_dim
        meta_in = CFG.num_genres + 1

        self.user_emb = nn.Embedding(n_users, d)
        self.item_emb = nn.Embedding(n_items, d)
        self.meta_enc = MetadataEncoder(meta_in, m, CFG.dropout)

        self.mlp = build_mlp(d + d + m, CFG.hidden_dims, 1, CFG.dropout)

        self._init_weights()

    def _init_weights(self):
        nn.init.normal_(self.user_emb.weight, std=0.01)
        nn.init.normal_(self.item_emb.weight, std=0.01)

    def forward(self, user_idx, item_idx, metadata):
        u = self.user_emb(user_idx)
        v = self.item_emb(item_idx)
        m = self.meta_enc(metadata)
        x = torch.cat([u, v, m], dim=-1)
        return self.mlp(x).squeeze(-1)


class MCF_NoMeta(nn.Module):
    """
    Многокритериальная модель без метаданных.
    UserEmb + ItemEmb → 4 головы критериев + фиксированные равные веса.
    """

    def __init__(self, n_users: int, n_items: int):
        super().__init__()
        d = CFG.embed_dim
        n_criteria = len(CFG.criteria)

        self.user_emb = nn.Embedding(n_users, d)
        self.item_emb = nn.Embedding(n_items, d)

        self.criterion_heads = nn.ModuleList(
            [
                build_mlp(d + d, CFG.hidden_dims, 1, CFG.dropout)
                for _ in range(n_criteria)
            ]
        )

        nn.init.normal_(self.user_emb.weight, std=0.01)
        nn.init.normal_(self.item_emb.weight, std=0.01)

    def forward(self, user_idx, item_idx, metadata=None):
        u = self.user_emb(user_idx)
        v = self.item_emb(item_idx)
        x = torch.cat([u, v], dim=-1)

        criterion_scores = torch.stack(
            [head(x).squeeze(-1) for head in self.criterion_heads], dim=1
        )

        weights = torch.ones_like(criterion_scores) / len(CFG.criteria)
        overall = (weights * criterion_scores).sum(dim=1)

        return overall, criterion_scores


class MCF_FixedWeights(nn.Module):
    """
    Многокритериальная модель с метаданными, но без динамических весов.
    Фиксированные равные веса критериев.
    """

    def __init__(self, n_users: int, n_items: int):
        super().__init__()
        d = CFG.embed_dim
        m = CFG.metadata_dim
        meta_in = CFG.num_genres + 1
        n_criteria = len(CFG.criteria)

        self.user_emb = nn.Embedding(n_users, d)
        self.item_emb = nn.Embedding(n_items, d)
        self.meta_enc = MetadataEncoder(meta_in, m, CFG.dropout)

        self.criterion_heads = nn.ModuleList(
            [
                build_mlp(d + d + m, CFG.hidden_dims, 1, CFG.dropout)
                for _ in range(n_criteria)
            ]
        )

        nn.init.normal_(self.user_emb.weight, std=0.01)
        nn.init.normal_(self.item_emb.weight, std=0.01)

    def forward(self, user_idx, item_idx, metadata):
        u = self.user_emb(user_idx)
        v = self.item_emb(item_idx)
        m = self.meta_enc(metadata)
        x = torch.cat([u, v, m], dim=-1)

        criterion_scores = torch.stack(
            [head(x).squeeze(-1) for head in self.criterion_heads], dim=1
        )

        weights = torch.ones_like(criterion_scores) / len(CFG.criteria)
        overall = (weights * criterion_scores).sum(dim=1)

        return overall, criterion_scores


class MCF_Full(nn.Module):
    """
    Полная многокритериальная модель с динамическими персональными весами.
    UserWeightHead(user_emb, genre_emb) → softmax → [w1..w4].
    Веса зависят от жанровых предпочтений пользователя (новизна).
    """

    def __init__(self, n_users: int, n_items: int):
        super().__init__()
        d = CFG.embed_dim
        m = CFG.metadata_dim
        meta_in = CFG.num_genres + 1
        n_criteria = len(CFG.criteria)

        self.user_emb = nn.Embedding(n_users, d)
        self.item_emb = nn.Embedding(n_items, d)
        self.user_bias = nn.Embedding(n_users, 1)
        self.item_bias = nn.Embedding(n_items, 1)

        self.meta_enc = MetadataEncoder(meta_in, m, CFG.dropout)
        self.genre_enc = nn.Sequential(nn.Linear(CFG.num_genres, 32), nn.ReLU())

        self.weight_head = nn.Sequential(
            nn.Linear(d + 32, 128),
            nn.ReLU(),
            nn.Dropout(CFG.dropout),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, n_criteria),
        )

        self.criterion_heads = nn.ModuleList(
            [
                build_mlp(d + d + m, CFG.hidden_dims, 1, CFG.dropout)
                for _ in range(n_criteria)
            ]
        )

        self.direct_head = build_mlp(d + d + m, CFG.hidden_dims, 1, CFG.dropout)
        self.blend_logit = nn.Parameter(torch.zeros(1))

        nn.init.normal_(self.user_emb.weight, std=0.01)
        nn.init.normal_(self.item_emb.weight, std=0.01)
        nn.init.zeros_(self.user_bias.weight)
        nn.init.zeros_(self.item_bias.weight)

    def forward(self, user_idx, item_idx, metadata):
        u = self.user_emb(user_idx)
        v = self.item_emb(item_idx)
        m = self.meta_enc(metadata)
        x = torch.cat([u, v, m], dim=-1)

        genre_profile = self.genre_enc(metadata[:, 1:])
        weights = torch.softmax(
            self.weight_head(torch.cat([u, genre_profile], dim=-1)), dim=-1
        )

        criterion_scores = torch.stack(
            [head(x).squeeze(-1) for head in self.criterion_heads], dim=1
        )

        criteria_overall = (weights * criterion_scores).sum(dim=1)
        direct_overall = self.direct_head(x).squeeze(-1)

        alpha = torch.sigmoid(self.blend_logit)
        bias = self.user_bias(user_idx).squeeze(-1) + self.item_bias(item_idx).squeeze(
            -1
        )
        overall = alpha * criteria_overall + (1 - alpha) * direct_overall + bias

        return overall, criterion_scores, weights
