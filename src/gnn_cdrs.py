"""LightGCN-style CDRS for rating prediction (user-overlap transfer).

Two-phase protocol matched to MF/MFB/NMF/SVD/DMF:
  1) Train LightGCN on dense source bipartite graph (MSE on observed ratings).
  2) Train on sparse target with user-embedding transfer (gamma pull to source users).

No torch-geometric dependency — pure PyTorch index-based propagation.
"""

from __future__ import annotations

from typing import Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


def _to_numpy(R) -> np.ndarray:
    return R.to_numpy() if hasattr(R, "to_numpy") else np.asarray(R)


def observed_triples(R: np.ndarray) -> np.ndarray:
    u, i = np.where(R > 0)
    return np.column_stack([u, i, R[u, i]]).astype(np.float64)


def evaluate(R_true: np.ndarray, R_pred: np.ndarray) -> Tuple[float, float]:
    mask = R_true > 0
    err = R_pred[mask] - R_true[mask]
    return float(np.mean(np.abs(err))), float(np.sqrt(np.mean(err**2)))


def _device(prefer_cuda: bool = True) -> torch.device:
    if prefer_cuda and torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


class LightGCNRating(nn.Module):
    """Bipartite LightGCN + biases for explicit rating prediction."""

    def __init__(
        self,
        n_users: int,
        n_items: int,
        n_factors: int = 10,
        n_layers: int = 2,
        seed: int = 42,
    ):
        super().__init__()
        self.n_users = n_users
        self.n_items = n_items
        self.n_layers = n_layers
        g = torch.Generator()
        g.manual_seed(seed)
        self.user_emb = nn.Embedding(n_users, n_factors)
        self.item_emb = nn.Embedding(n_items, n_factors)
        nn.init.normal_(self.user_emb.weight, std=0.1, generator=g)
        nn.init.normal_(self.item_emb.weight, std=0.1, generator=g)
        self.user_bias = nn.Embedding(n_users, 1)
        self.item_bias = nn.Embedding(n_items, 1)
        nn.init.zeros_(self.user_bias.weight)
        nn.init.zeros_(self.item_bias.weight)
        self.global_bias = nn.Parameter(torch.zeros(1))

        # Graph buffers filled by set_graph
        self.register_buffer("edge_u", torch.empty(0, dtype=torch.long), persistent=False)
        self.register_buffer("edge_i", torch.empty(0, dtype=torch.long), persistent=False)
        self.register_buffer("norm_ui", torch.empty(0), persistent=False)
        self.register_buffer("norm_iu", torch.empty(0), persistent=False)

    def set_graph(self, R: np.ndarray) -> None:
        """Build normalized bipartite edges from observed ratings (>0)."""
        u, i = np.where(R > 0)
        if len(u) == 0:
            # Dummy self-loop-free empty graph
            self.edge_u = torch.zeros(0, dtype=torch.long, device=self.user_emb.weight.device)
            self.edge_i = torch.zeros(0, dtype=torch.long, device=self.user_emb.weight.device)
            self.norm_ui = torch.zeros(0, device=self.user_emb.weight.device)
            self.norm_iu = torch.zeros(0, device=self.user_emb.weight.device)
            return
        deg_u = np.bincount(u, minlength=self.n_users).astype(np.float64)
        deg_i = np.bincount(i, minlength=self.n_items).astype(np.float64)
        # LightGCN: 1/sqrt(deg_u * deg_i)
        norm = 1.0 / np.sqrt(np.maximum(deg_u[u], 1.0) * np.maximum(deg_i[i], 1.0))
        device = self.user_emb.weight.device
        self.edge_u = torch.as_tensor(u, dtype=torch.long, device=device)
        self.edge_i = torch.as_tensor(i, dtype=torch.long, device=device)
        self.norm_ui = torch.as_tensor(norm, dtype=torch.float32, device=device)
        self.norm_iu = self.norm_ui  # same scalar per undirected edge

    def propagate(self) -> Tuple[torch.Tensor, torch.Tensor]:
        eu0 = self.user_emb.weight
        ei0 = self.item_emb.weight
        eu, ei = eu0, ei0
        eu_all = [eu]
        ei_all = [ei]
        for _ in range(self.n_layers):
            # users <- items
            msg_u = torch.zeros_like(eu)
            if self.edge_u.numel() > 0:
                src = ei[self.edge_i] * self.norm_ui.unsqueeze(1)
                msg_u.index_add_(0, self.edge_u, src)
            # items <- users
            msg_i = torch.zeros_like(ei)
            if self.edge_i.numel() > 0:
                src = eu[self.edge_u] * self.norm_iu.unsqueeze(1)
                msg_i.index_add_(0, self.edge_i, src)
            eu, ei = msg_u, msg_i
            eu_all.append(eu)
            ei_all.append(ei)
        eu_f = torch.stack(eu_all, dim=0).mean(dim=0)
        ei_f = torch.stack(ei_all, dim=0).mean(dim=0)
        return eu_f, ei_f

    def predict_batch(self, users: torch.Tensor, items: torch.Tensor) -> torch.Tensor:
        eu, ei = self.propagate()
        return (
            self.global_bias
            + self.user_bias(users).squeeze(-1)
            + self.item_bias(items).squeeze(-1)
            + (eu[users] * ei[items]).sum(dim=-1)
        )

    @torch.no_grad()
    def predict_observed(self, R_true: np.ndarray) -> Tuple[float, float]:
        """MAE/RMSE only on observed entries of R_true (avoids huge dense mats)."""
        u, i = np.where(R_true > 0)
        if len(u) == 0:
            return float("nan"), float("nan")
        device = self.user_emb.weight.device
        users = torch.as_tensor(u, dtype=torch.long, device=device)
        items = torch.as_tensor(i, dtype=torch.long, device=device)
        # chunk to limit peak memory on large Amazon pairs
        preds = []
        bs = 65536
        for s in range(0, len(u), bs):
            preds.append(self.predict_batch(users[s : s + bs], items[s : s + bs]))
        pred = torch.cat(preds).detach().cpu().numpy()
        true = R_true[u, i]
        err = pred - true
        return float(np.mean(np.abs(err))), float(np.sqrt(np.mean(err**2)))


def _train_phase(
    model: LightGCNRating,
    R: np.ndarray,
    *,
    epochs: int,
    lr: float,
    lambda_reg: float,
    batch_size: int,
    seed: int,
    source_user_emb: Optional[torch.Tensor] = None,
    gamma: float = 0.0,
) -> None:
    """One LightGCN propagate + one backward per epoch (fast on large Amazon graphs)."""
    device = next(model.parameters()).device
    model.set_graph(R)
    triples = observed_triples(R)
    if len(triples) == 0:
        return
    rng = np.random.default_rng(seed)
    opt = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=0.0)
    n = len(triples)
    mu = float(R[R > 0].mean()) if np.any(R > 0) else 0.0
    with torch.no_grad():
        model.global_bias.fill_(mu)

    users_all = torch.as_tensor(triples[:, 0], dtype=torch.long, device=device)
    items_all = torch.as_tensor(triples[:, 1], dtype=torch.long, device=device)
    ratings_all = torch.as_tensor(triples[:, 2], dtype=torch.float32, device=device)
    # Cap edges scored per epoch (large Amazon Book targets); still several Adam steps/epoch.
    epoch_cap = min(n, max(batch_size * 8, 65536))

    for _ep in range(epochs):
        pick = rng.choice(n, size=epoch_cap, replace=False)
        for start in range(0, epoch_cap, batch_size):
            idx = pick[start : start + batch_size]
            idx_t = torch.as_tensor(idx, dtype=torch.long, device=device)
            users = users_all[idx_t]
            items = items_all[idx_t]
            ratings = ratings_all[idx_t]
            eu_prop, ei_prop = model.propagate()
            pred = (
                model.global_bias
                + model.user_bias(users).squeeze(-1)
                + model.item_bias(items).squeeze(-1)
                + (eu_prop[users] * ei_prop[items]).sum(dim=-1)
            )
            loss = F.mse_loss(pred, ratings)
            loss = loss + lambda_reg * (
                model.user_emb(users).pow(2).mean() + model.item_emb(items).pow(2).mean()
            )
            if source_user_emb is not None and gamma > 0:
                uniq = torch.unique(users)
                loss = loss + gamma * F.mse_loss(model.user_emb(uniq), source_user_emb[uniq])
            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step()


def run_gnn_once(
    R_source,
    R_train,
    R_test,
    n_factors: int = 10,
    n_layers: int = 2,
    lambda_reg: float = 0.01,
    gamma: float = 0.1,
    learning_rate: float = 0.001,
    epochs: int = 20,
    batch_size: int = 2048,
    seed: int = 42,
    prefer_cuda: bool = True,
) -> Tuple[float, float]:
    """Source LightGCN → target transfer → MAE/RMSE on R_test."""
    R_source = _to_numpy(R_source).astype(np.float64)
    R_train = _to_numpy(R_train).astype(np.float64)
    R_test = _to_numpy(R_test).astype(np.float64)
    device = _device(prefer_cuda)
    n_users, n_items_s = R_source.shape
    n_items_t = R_train.shape[1]

    torch.manual_seed(seed)
    if device.type == "cuda":
        torch.cuda.manual_seed_all(seed)

    # Phase 1: source
    src = LightGCNRating(n_users, n_items_s, n_factors=n_factors, n_layers=n_layers, seed=seed).to(device)
    _train_phase(
        src,
        R_source,
        epochs=epochs,
        lr=learning_rate,
        lambda_reg=lambda_reg,
        batch_size=batch_size,
        seed=seed,
    )
    source_user = src.user_emb.weight.detach().clone()

    # Phase 2: target with user init + gamma transfer
    tgt = LightGCNRating(
        n_users, n_items_t, n_factors=n_factors, n_layers=n_layers, seed=seed + 1
    ).to(device)
    with torch.no_grad():
        n_share = min(n_users, source_user.shape[0])
        tgt.user_emb.weight[:n_share].copy_(source_user[:n_share])
        tgt.user_bias.weight[:n_share].copy_(src.user_bias.weight[:n_share])
        tgt.global_bias.copy_(src.global_bias)

    _train_phase(
        tgt,
        R_train,
        epochs=epochs,
        lr=learning_rate,
        lambda_reg=lambda_reg,
        batch_size=batch_size,
        seed=seed + 2,
        source_user_emb=source_user,
        gamma=gamma,
    )
    return tgt.predict_observed(R_test)
