from __future__ import annotations

"""
prepara_dados.py
----------------
Calcula as features quantitativas relativas ao benchmark (Ibovespa) e
grava o resultado em cima do próprio CSV de cada ativo em dados_diarios/.

Notação:
  A  = ativo alvo
  M  = benchmark (Ibovespa)
  N  = janela móvel em dias úteis (configurável em WINDOWS)

Grupos implementados
────────────────────
  1. Momentum Relativo   : spread_retorno, dist_sma
  2. Sensibilidade/Risco : spread_vol, beta
  3. Microestrutura      : spread_volratio
  4. Excesso Estatístico : z_alpha
  5. Sentimento (NLP)    : requer pipeline externo — não implementado aqui.
  6. Regime de Mercado   : trend_mercado, risco_mercado, z_mercado

Target (sem look-ahead bias)
─────────────────────────────
  Y_{t+1} = R_A,t+1 - R_M,t+1  (alpha: excesso de retorno sobre o Ibovespa)
  Todas as features são calculadas com dados estritamente até t.
"""

import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT     = Path(__file__).resolve().parent
DATA_DIR = ROOT / "dados_diarios"
IBOV_FILE = DATA_DIR / "IBOV.csv"

# Janelas móveis em dias úteis — três horizontes macro:
#   Microestrutura  [3, 5]      → absorção de informação / choques de volume
#   Ciclo Mensal    [14, 21]    → risco tático, beta, osciladores
#   Regime Macro    [42, 63]    → tendência primária, ciclo de earnings
WINDOWS: list[int] = [3, 5, 14, 21, 42, 63]


# ─── helpers ─────────────────────────────────────────────────────────────────

def log_ret(s: pd.Series) -> pd.Series:
    return np.log(s / s.shift(1))


def rolling_beta(r_a: pd.Series, r_m: pd.Series, n: int) -> pd.Series:
    """Beta móvel: Cov(R_A, R_M, N) / Var(R_M, N)."""
    cov = r_a.rolling(n, min_periods=n).cov(r_m)
    var = r_m.rolling(n, min_periods=n).var(ddof=1)
    return cov / var.replace(0, np.nan)


# ─── engenharia de features ───────────────────────────────────────────────────

def calcular_features(df: pd.DataFrame, ibov: pd.DataFrame, n: int) -> pd.DataFrame:
    """
    Calcula todas as features para a janela N e adiciona colunas ao DataFrame.
    df e ibov devem estar alinhados por data (mesmo índice).
    """
    p_a = df["close"]
    p_m = ibov["close"]
    v_a = df["volume"].astype(float)
    v_m = ibov["volume"].astype(float)

    r_a = log_ret(p_a)
    r_m = log_ret(p_m)

    sfx = f"_{n}d"

    # ── retornos diários (salvos apenas na primeira janela) ───────────────
    if f"log_ret" not in df.columns:
        df["log_ret"]      = r_a.values
        df["ibov_log_ret"] = r_m.values

    # ── 1. Momentum Relativo ──────────────────────────────────────────────

    # Spread de Retorno Acumulado: ln(P_A,t / P_A,t-N) - ln(P_M,t / P_M,t-N)
    df[f"spread_retorno{sfx}"] = (
        np.log(p_a / p_a.shift(n)) - np.log(p_m / p_m.shift(n))
    )

    # Distância Relativa da Média Móvel: (P_A/SMA_A - 1) - (P_M/SMA_M - 1)
    sma_a = p_a.rolling(n, min_periods=n).mean()
    sma_m = p_m.rolling(n, min_periods=n).mean()
    df[f"dist_sma{sfx}"] = (
        (p_a / sma_a.replace(0, np.nan) - 1.0) -
        (p_m / sma_m.replace(0, np.nan) - 1.0)
    )

    # ── 2. Sensibilidade e Risco ──────────────────────────────────────────

    # Spread de Volatilidade Histórica: σ(R_A, N) - σ(R_M, N)
    df[f"spread_vol{sfx}"] = (
        r_a.rolling(n, min_periods=n).std(ddof=1) -
        r_m.rolling(n, min_periods=n).std(ddof=1)
    )

    # Beta Móvel: Cov(R_A, R_M, N) / Var(R_M, N)
    df[f"beta{sfx}"] = rolling_beta(r_a, r_m, n)

    # ── 3. Microestrutura e Liquidez ──────────────────────────────────────

    # Spread de Volume Ratio: (V_A / μ_A) - (V_M / μ_M)
    mu_va = v_a.rolling(n, min_periods=n).mean()
    mu_vm = v_m.rolling(n, min_periods=n).mean()
    df[f"spread_volratio{sfx}"] = (
        v_a / mu_va.replace(0, np.nan) -
        v_m / mu_vm.replace(0, np.nan)
    )

    # ── 4. Excesso Estatístico ────────────────────────────────────────────

    # Z-Score do Alpha Diário: (R_A - R_M - μ_alpha) / σ_alpha
    alpha = r_a - r_m
    mu_alpha  = alpha.rolling(n, min_periods=n).mean()
    std_alpha = alpha.rolling(n, min_periods=n).std(ddof=1)
    df[f"z_alpha{sfx}"] = (alpha - mu_alpha) / std_alpha.replace(0, np.nan)

    # ── 6. Regime de Mercado (Macro Contexto) ────────────────────────────
    # Usa apenas dados do Ibovespa — nenhuma informação futura do ativo.

    # Direção da Tendência: ln(P_M,t / P_M,t-N)
    df[f"trend_mercado{sfx}"] = np.log(p_m / p_m.shift(n))

    # Temperatura de Risco: σ(R_M, N) — volatilidade realizada do índice
    df[f"risco_mercado{sfx}"] = r_m.rolling(n, min_periods=n).std(ddof=1)

    # Distância Histórica: (P_M,t - μ(P_M, N)) / σ(P_M, N)
    mu_pm  = p_m.rolling(n, min_periods=n).mean()
    std_pm = p_m.rolling(n, min_periods=n).std(ddof=1)
    df[f"z_mercado{sfx}"] = (p_m - mu_pm) / std_pm.replace(0, np.nan)

    return df


# ─── pipeline por ticker ─────────────────────────────────────────────────────

def processar_ticker(fp: Path, ibov: pd.DataFrame) -> None:
    ticker = fp.stem
    print(f"  [{ticker}] calculando features ...", end=" ", flush=True)

    df = pd.read_csv(fp, parse_dates=["date"])
    df = df.sort_values("date").reset_index(drop=True)

    # alinha o Ibovespa ao calendário do ativo (forward-fill para feriados locais)
    ibov_alinhado = (
        df[["date"]]
        .merge(ibov[["date", "close", "volume"]], on="date", how="left")
        .set_index("date")
    )
    ibov_alinhado = ibov_alinhado.ffill()
    ibov_alinhado = ibov_alinhado.reset_index()

    # persiste ibov_close no CSV para uso do treina_gru.py (cálculo do alpha)
    df["ibov_close"] = ibov_alinhado["close"].values

    # remove colunas de features antigas para recalcular
    feature_prefixes = ("spread_retorno", "dist_sma", "spread_vol",
                        "beta", "spread_volratio", "z_alpha",
                        "trend_mercado", "risco_mercado", "z_mercado")
    df = df.drop(columns=[c for c in df.columns
                           if any(c.startswith(p) for p in feature_prefixes)],
                 errors="ignore")

    for n in WINDOWS:
        df = calcular_features(df, ibov_alinhado, n)

    df.to_csv(fp, index=False)

    n_features = len(WINDOWS) * 9   # 9 features × 6 janelas = 54 features
    print(f"ok — {len(df)} dias | {n_features} features ({len(WINDOWS)} janela(s))")


# ─── main ────────────────────────────────────────────────────────────────────

def main() -> None:
    if not IBOV_FILE.exists():
        print(f"Arquivo do Ibovespa não encontrado: {IBOV_FILE}")
        print("Rode novo_input_dados.py primeiro.")
        sys.exit(1)

    ibov = pd.read_csv(IBOV_FILE, parse_dates=["date"])
    ibov = ibov.sort_values("date").reset_index(drop=True)
    print(f"[IBOV] {len(ibov)} dias carregados\n")

    # exclui o próprio arquivo do Ibovespa da lista de tickers
    csvs = sorted(p for p in DATA_DIR.glob("*.csv") if p.stem != "IBOV")
    if not csvs:
        print(f"Nenhum CSV de ativo em {DATA_DIR}. Rode novo_input_dados.py primeiro.")
        sys.exit(1)

    print(f"Processando {len(csvs)} tickers")
    print(f"Janelas: {WINDOWS} dias ({len(WINDOWS) * 9} features por ativo)\n")
    erros: list[tuple[str, str]] = []

    for fp in csvs:
        try:
            processar_ticker(fp, ibov)
        except Exception as exc:
            erros.append((fp.stem, str(exc)))
            print(f"ERRO: {exc}")

    print("\n── Features geradas ─────────────────────────────────────────────")
    horizons = {
        "Microestrutura [3, 5]"   : [n for n in WINDOWS if n <= 5],
        "Ciclo Mensal   [14, 21]" : [n for n in WINDOWS if 5 < n <= 21],
        "Regime Macro   [42, 63]" : [n for n in WINDOWS if n > 21],
    }
    features_por_grupo = [
        ("1. Momentum",    "spread_retorno, dist_sma"),
        ("2. Risco",       "spread_vol, beta"),
        ("3. Liquidez",    "spread_volratio"),
        ("4. Estatístico", "z_alpha"),
        ("6. Regime",      "trend_mercado, risco_mercado, z_mercado"),
    ]
    for horizonte, ns in horizons.items():
        print(f"  {horizonte}: janelas {ns}d")
    print(f"\n  Features por janela:")
    for label, cols in features_por_grupo:
        print(f"    {label}: {cols}")
    print(f"\n  Total: {len(WINDOWS)} janelas × 9 features = {len(WINDOWS) * 9} features")
    print("  Grupo 5 (Sentimento NLP): pendente de pipeline externo.")
    print("\n  Target: Y_{{t+1}} = R_A,t+1 - R_M,t+1  (alpha sobre o Ibovespa)")

    if erros:
        print("\nErros:")
        for t, msg in erros:
            print(f"  {t}: {msg}")
    else:
        print(f"\nTodos os {len(csvs)} tickers processados com sucesso.")


if __name__ == "__main__":
    main()
