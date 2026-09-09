"""Parâmetros tunáveis do pipeline Black-Litterman."""
from __future__ import annotations

from dataclasses import asdict, dataclass

try:
    from .config import (
        ALPHA_Q,
        CONFIDENCE_CAP,
        CONFIDENCE_FLOOR,
        ERR_MIN_PERIODS,
        ERR_WINDOW,
        LONG_ONLY,
        MAX_WEIGHT_PER_ASSET,
        ROLLING_MIN_OBS,
        TAU,
        USE_ROLLING_PRIOR,
        W_MODEL_CONF,
        W_NEWS_CONF,
    )
except ImportError:
    from config import (
        ALPHA_Q,
        CONFIDENCE_CAP,
        CONFIDENCE_FLOOR,
        ERR_MIN_PERIODS,
        ERR_WINDOW,
        LONG_ONLY,
        MAX_WEIGHT_PER_ASSET,
        ROLLING_MIN_OBS,
        TAU,
        USE_ROLLING_PRIOR,
        W_MODEL_CONF,
        W_NEWS_CONF,
    )

__all__ = ["BLParams"]


@dataclass(frozen=True)
class BLParams:
    """Hiperparâmetros do BL híbrido (valores padrão = ``config.py``)."""

    alpha_q: float = ALPHA_Q
    tau: float = TAU
    max_weight_per_asset: float = MAX_WEIGHT_PER_ASSET
    w_model_conf: float = W_MODEL_CONF
    w_news_conf: float = W_NEWS_CONF
    confidence_floor: float = CONFIDENCE_FLOOR
    confidence_cap: float = CONFIDENCE_CAP
    err_window: int = ERR_WINDOW
    err_min_periods: int = ERR_MIN_PERIODS
    rolling_min_obs: int = ROLLING_MIN_OBS
    use_rolling_prior: bool = USE_ROLLING_PRIOR
    long_only: bool = LONG_ONLY
    rebalance_mode: str = "weekly"

    @classmethod
    def from_config(cls, rebalance_mode: str = "weekly") -> BLParams:
        return cls(rebalance_mode=rebalance_mode)

    def normalized(self) -> BLParams:
        """Garante pesos de confiança e limites consistentes."""
        w_model = float(min(max(self.w_model_conf, 0.0), 1.0))
        w_news = float(min(max(self.w_news_conf, 0.0), 1.0))
        total = w_model + w_news
        if total <= 0:
            w_model, w_news = 0.7, 0.3
        else:
            w_model, w_news = w_model / total, w_news / total

        floor = float(min(self.confidence_floor, self.confidence_cap))
        cap = float(max(self.confidence_floor, self.confidence_cap))
        err_window = max(int(self.err_window), 2)
        err_min_periods = min(max(int(self.err_min_periods), 2), err_window)

        return BLParams(
            alpha_q=float(min(max(self.alpha_q, 0.0), 1.0)),
            tau=max(float(self.tau), 1e-6),
            max_weight_per_asset=float(min(max(self.max_weight_per_asset, 0.05), 1.0)),
            w_model_conf=w_model,
            w_news_conf=w_news,
            confidence_floor=floor,
            confidence_cap=cap,
            err_window=err_window,
            err_min_periods=err_min_periods,
            rolling_min_obs=max(int(self.rolling_min_obs), 20),
            use_rolling_prior=self.use_rolling_prior,
            long_only=self.long_only,
            rebalance_mode=self.rebalance_mode,
        )

    def to_dict(self) -> dict[str, float | int | bool | str]:
        return asdict(self.normalized())
