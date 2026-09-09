"""Configuração central do pipeline Black-Litterman."""
from __future__ import annotations

from pathlib import Path

BL_DIR = Path(__file__).resolve().parent
REPO_ROOT = BL_DIR.parents[1]

DATA_DAILY_DIR = REPO_ROOT / "dados_diarios"
XGB_DIR = REPO_ROOT / "criacao_modelo_xgb"
NEWS_PATH = REPO_ROOT / "intel" / "data_acquisition" / "noticias_b3_sem_duplicados_sentimento.json"

# Diretórios de saída
OUTPUTS_DIR = BL_DIR / "outputs"
PIPELINE_DIR = OUTPUTS_DIR / "pipeline"
BACKTEST_DIR = OUTPUTS_DIR / "backtest"
SCORE_DIR = OUTPUTS_DIR / "score"
MARKOWITZ_DIR = OUTPUTS_DIR / "markowitz"
FIGURES_DIR = OUTPUTS_DIR / "figures"
OPTIMIZE_DIR = OUTPUTS_DIR / "optimize"
STANDALONE_Q_DIR = OUTPUTS_DIR / "standalone" / "q"
STANDALONE_OMEGA_DIR = OUTPUTS_DIR / "standalone" / "omega"
STANDALONE_PRIOR_DIR = OUTPUTS_DIR / "standalone" / "prior"

for _dir in (
    PIPELINE_DIR,
    BACKTEST_DIR,
    SCORE_DIR,
    MARKOWITZ_DIR,
    FIGURES_DIR,
    OPTIMIZE_DIR,
    STANDALONE_Q_DIR,
    STANDALONE_OMEGA_DIR,
    STANDALONE_PRIOR_DIR,
):
    _dir.mkdir(parents=True, exist_ok=True)

# Views (Q)
SENTIMENT_MAP = {"positivo": 1.0, "negativo": -1.0, "neutro": 0.0}
PERIOD_START = "202501"
PERIOD_END = "202512"
ALPHA_Q = 0.343669
RET_SCALE_WINDOW = 14
RET_SCALE_MIN_PERIODS = 5
RET_SCALE_FALLBACK = 0.005

# Prior (PI)
ATIVOS = ["VALE3.SA", "BBAS3.SA", "ITUB4.SA", "BBDC4.SA", "ABEV3.SA"]
PRIOR_START_DATE = "2023-01-01"
PRIOR_END_DATE = "2025-12-31"
RISK_FREE_ANNUAL = 0.15
TAU = 0.0213796
USE_ROLLING_PRIOR = True
ROLLING_MIN_OBS = 60

# Omega dinâmico
ERR_WINDOW = 27
ERR_MIN_PERIODS = 4
EPS = 1e-8
CONFIDENCE_FLOOR = 0.192838
CONFIDENCE_CAP = 0.917401
W_MODEL_CONF = 0.314511
W_NEWS_CONF = 0.685489

# Controles de portfólio
LONG_ONLY = True
MAX_WEIGHT_PER_ASSET = 0.494293

# Otimização de hiperparâmetros (treino / validação out-of-sample)
OPT_TRAIN_START = "202501"
OPT_TRAIN_END = "202506"
OPT_VAL_START = "202507"
OPT_VAL_END = "202512"
OPT_DEFAULT_TRIALS = 40
OPT_DEFAULT_REBALANCE_MODE = "weekly"

# Saídas do pipeline híbrido → outputs/pipeline/
OUT_Q_LONG = PIPELINE_DIR / "bl_hibrido_q_long.csv"
OUT_OMEGA_LONG = PIPELINE_DIR / "bl_hibrido_omega_long.csv"
OUT_Q_OMEGA_LONG = PIPELINE_DIR / "bl_hibrido_q_omega_long.csv"
OUT_PI = PIPELINE_DIR / "bl_hibrido_pi.csv"
OUT_SIGMA = PIPELINE_DIR / "bl_hibrido_sigma.csv"
OUT_PRIOR_COV = PIPELINE_DIR / "bl_hibrido_prior_cov_tau_sigma.csv"
OUT_POSTERIOR_MU = PIPELINE_DIR / "bl_hibrido_posterior_mu.csv"
OUT_POSTERIOR_W = PIPELINE_DIR / "bl_hibrido_posterior_weights.csv"
OUT_POSTERIOR_W_CONTROLLED = PIPELINE_DIR / "bl_hibrido_posterior_weights_controlled.csv"
OUT_POSTERIOR_W_CTRL_DAILY = PIPELINE_DIR / "bl_hibrido_posterior_weights_controlled_daily.csv"
OUT_POSTERIOR_W_CTRL_WEEKLY = PIPELINE_DIR / "bl_hibrido_posterior_weights_controlled_weekly.csv"
OUT_POSTERIOR_W_CTRL_MONTHLY = PIPELINE_DIR / "bl_hibrido_posterior_weights_controlled_monthly.csv"

# Saídas do Q standalone → outputs/standalone/q/
OUT_Q_STANDALONE_LONG = STANDALONE_Q_DIR / "q_walkforward_long.csv"
OUT_Q_STANDALONE_MATRIX = STANDALONE_Q_DIR / "q_walkforward_matrix.csv"

# Avaliação / backtest
INITIAL_CAPITAL_BRL = 100_000.0
SELIC_ANNUAL = 0.15
EXECUTION_LAG_DAYS = 1
TRADING_DAYS_YEAR = 252

# Backtest → outputs/backtest/
OUT_GAIN_SERIES_DAILY = BACKTEST_DIR / "bl_hibrido_ganho_series_daily.csv"
OUT_GAIN_SERIES_WEEKLY = BACKTEST_DIR / "bl_hibrido_ganho_series_weekly.csv"
OUT_GAIN_SERIES_MONTHLY = BACKTEST_DIR / "bl_hibrido_ganho_series_monthly.csv"
OUT_GAIN_COMPARATIVE = BACKTEST_DIR / "bl_hibrido_ganho_comparativo.csv"
OUT_GAIN_FALLBACK = BACKTEST_DIR / "bl_hibrido_ganho_series.csv"

# Figuras → outputs/figures/
OUT_GAIN_PLOT = FIGURES_DIR / "bl_hibrido_ganho_estimado.png"
OUT_MKZ_PLOT = FIGURES_DIR / "bl_vs_markowitz_plot.png"

# Score → outputs/score/
OUT_SCORE_SUMMARY = SCORE_DIR / "bl_hibrido_score_summary.csv"
OUT_SCORE_MONTHLY = SCORE_DIR / "bl_hibrido_score_mensal.csv"
OUT_SCORE_REBALANCE = SCORE_DIR / "bl_hibrido_rebalance_resumo.csv"
OUT_SCORE_PRED_METRICS = SCORE_DIR / "bl_hibrido_score_previsao_metricas.csv"
OUT_SCORE_PRED_TICKER = SCORE_DIR / "bl_hibrido_score_previsao_por_ticker.csv"
OUT_SCORE_PRED_MONTH = SCORE_DIR / "bl_hibrido_score_previsao_por_mes.csv"

# Markowitz → outputs/markowitz/
OUT_MKZ_COMP = MARKOWITZ_DIR / "bl_vs_markowitz_comparativo.csv"
OUT_MKZ_SUMMARY = MARKOWITZ_DIR / "bl_vs_markowitz_resumo.csv"
OUT_MKZ_W_DAILY = MARKOWITZ_DIR / "markowitz_weights_daily.csv"
OUT_MKZ_W_WEEKLY = MARKOWITZ_DIR / "markowitz_weights_weekly.csv"
OUT_MKZ_W_MONTHLY = MARKOWITZ_DIR / "markowitz_weights_monthly.csv"

# Omega standalone → outputs/standalone/omega/
OUT_OMEGA_STANDALONE_LONG = STANDALONE_OMEGA_DIR / "omega_walkforward_long.csv"
OUT_OMEGA_STANDALONE_MATRIX = STANDALONE_OMEGA_DIR / "omega_walkforward_matrix.csv"
OUT_Q_OMEGA_STANDALONE_LONG = STANDALONE_OMEGA_DIR / "q_omega_walkforward_long.csv"
OUT_Q_OMEGA_STANDALONE_WIDE = STANDALONE_OMEGA_DIR / "q_omega_walkforward_wide.csv"

# PI standalone (VWAP) → outputs/standalone/prior/
OUT_PI_STANDALONE = STANDALONE_PRIOR_DIR / "pi_equilibrio.csv"
OUT_SIGMA_STANDALONE = STANDALONE_PRIOR_DIR / "sigma_cov.csv"
OUT_PRIOR_COV_STANDALONE = STANDALONE_PRIOR_DIR / "prior_cov_tau_sigma.csv"
OUT_PRIOR_DIAG_STANDALONE = STANDALONE_PRIOR_DIR / "prior_uncertainty_diag.csv"

# Otimização → outputs/optimize/
OUT_OPT_BEST_PARAMS = OPTIMIZE_DIR / "best_params.csv"
OUT_OPT_TRIALS = OPTIMIZE_DIR / "trials.csv"
OUT_OPT_SUMMARY = OPTIMIZE_DIR / "optimization_summary.json"
