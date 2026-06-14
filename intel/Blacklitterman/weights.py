"""Utilitários de alocação de portfólio (projeção long-only com teto por ativo)."""
from __future__ import annotations

import numpy as np

try:
    from .config import EPS
except ImportError:
    from config import EPS

__all__ = ["long_only_capped_weights"]


def long_only_capped_weights(raw_w: np.ndarray, max_w: float) -> np.ndarray:
    """
    Projeta pesos brutos em simplex long-only com teto ``max_w`` por ativo.

    Usa alocação iterativa: satura ativos que excedem o teto e redistribui o
    orçamento restante proporcionalmente aos scores positivos.
    """
    n = len(raw_w)
    if n == 0:
        return raw_w

    if max_w <= 0 or max_w * n < 1.0:
        raise ValueError(f"max_w={max_w} inviável para {n} ativos (precisa max_w*n >= 1).")

    scores = np.maximum(raw_w.astype(float), 0.0)
    if not np.isfinite(scores).all() or scores.sum() <= EPS:
        return np.full(n, 1.0 / n, dtype=float)

    w = np.zeros(n, dtype=float)
    remaining = np.ones(n, dtype=bool)
    budget = 1.0

    while budget > EPS and np.any(remaining):
        idx = np.where(remaining)[0]
        s = scores[idx]
        s_sum = float(s.sum())
        if s_sum <= EPS:
            w[idx] += budget / len(idx)
            break

        proposal = budget * (s / s_sum)
        saturated = proposal >= max_w - EPS

        if not np.any(saturated):
            w[idx] += proposal
            break

        sat_idx = idx[saturated]
        for j in sat_idx:
            alloc = max_w - w[j]
            if alloc > 0:
                w[j] += alloc
                budget -= alloc
            remaining[j] = False

    w = np.clip(w, 0.0, max_w)
    total = float(w.sum())
    return w / total if total > EPS else np.full(n, 1.0 / n, dtype=float)
