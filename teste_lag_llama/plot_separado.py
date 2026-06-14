from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parent
SAMPLES_PATH = PROJECT_ROOT / "teste_lagllama_amostras_diarias.csv"
OUT_PLOT = PROJECT_ROOT / "plot_separado_fan_maioria.png"
OUT_SERIES = PROJECT_ROOT / "plot_separado_fan_maioria_series.csv"
OUT_SCORE = PROJECT_ROOT / "plot_separado_fan_maioria_score.csv"
OUT_STRATEGY = PROJECT_ROOT / "plot_separado_fan_maioria_strategy.csv"
OUT_RULE_PLOT = PROJECT_ROOT / "plot_separado_fan_regras_comparacao.png"

# Janela de visualizacao
PLOT_LAST_N_DAYS = 90

INVERT_PREDICTION = True
EPS = 1e-12
CONFIDENCE_THRESHOLD = 0.20
QUANTILE_POS = 0.25
QUANTILE_NEG = 0.75
EXECUTION_LAG_DAYS = 1
TRANSACTION_COST_BPS = 0.0
ANNUALIZATION_DAILY = 252.0


def load_samples(path: Path) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(f"Arquivo de amostras nao encontrado: {path}")

    df = pd.read_csv(path)
    required = {"time_hour", "pred_sample", "real_value"}
    missing = required.difference(df.columns)
    if missing:
        raise ValueError(f"Colunas ausentes no CSV: {sorted(missing)}")

    df["time_hour"] = pd.to_datetime(df["time_hour"], errors="coerce")
    df["pred_sample"] = pd.to_numeric(df["pred_sample"], errors="coerce")
    df["real_value"] = pd.to_numeric(df["real_value"], errors="coerce")
    df = df.dropna(subset=["time_hour", "pred_sample", "real_value"]).copy()
    return df.sort_values("time_hour").reset_index(drop=True)


def filter_last_n_days(df: pd.DataFrame, n_days: int) -> pd.DataFrame:
    if df.empty:
        return df
    max_day = pd.Timestamp(df["time_hour"].max()).normalize()
    min_day = max_day - pd.Timedelta(days=n_days - 1)
    out = df[df["time_hour"].dt.normalize().between(min_day, max_day)].copy()
    return out.sort_values("time_hour").reset_index(drop=True)


def build_hourly_series(df: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict] = []
    for t, g in df.groupby("time_hour", sort=True):
        s = g["pred_sample"].to_numpy(dtype=float)
        if INVERT_PREDICTION:
            s = -s
        real = float(g["real_value"].iloc[0])

        q10, q25, q75, q90 = np.quantile(s, [0.10, 0.25, 0.75, 0.90])
        n_down = int(np.sum(s < 0.0))
        n_up = int(np.sum(s > 0.0))
        n_zero = int(np.sum(s == 0.0))
        n_nonzero = max(1, n_down + n_up)
        conf = float(abs(n_up - n_down) / n_nonzero)

        pos = s[s > 0.0]
        neg = s[s < 0.0]
        if n_down > n_up and n_down > 0:
            dir_sign = -1.0
            # Regra A: maioria + max/min (como esta hoje).
            pred_maxmin = float(np.max(neg))
            # Regra B: quantil no lado vencedor.
            pred_quantile = float(np.quantile(neg, QUANTILE_NEG))
        elif n_up > n_down and n_up > 0:
            dir_sign = 1.0
            pred_maxmin = float(np.min(pos))
            pred_quantile = float(np.quantile(pos, QUANTILE_POS))
        else:
            dir_sign = 0.0
            pred_maxmin = 0.0
            pred_quantile = 0.0

        # Regra C: abstem se confianca baixa.
        pred_conf_gate = pred_quantile if conf >= CONFIDENCE_THRESHOLD else 0.0
        # Regra D: mesmo gate + magnitude ponderada por confianca.
        pred_conf_weighted = pred_conf_gate * conf

        rows.append(
            {
                "time_hour": t,
                "real_value": real,
                "pred_maxmin": pred_maxmin,
                "pred_quantile": pred_quantile,
                "pred_conf_gate": pred_conf_gate,
                "pred_conf_weighted": pred_conf_weighted,
                "majority_direction": "down" if dir_sign < 0 else ("up" if dir_sign > 0 else "tie"),
                "direction_sign": dir_sign,
                "confidence": conf,
                "n_down": n_down,
                "n_up": n_up,
                "n_zero": n_zero,
                "pred_q10": float(q10),
                "pred_q25": float(q25),
                "pred_q75": float(q75),
                "pred_q90": float(q90),
            }
        )

    out = pd.DataFrame(rows).sort_values("time_hour").reset_index(drop=True)
    return out


def plot_real_vs_majority_with_smooth_fog(series_df: pd.DataFrame, out_path: Path) -> None:
    x = series_df["time_hour"]

    plt.figure(figsize=(14, 6))
    plt.fill_between(x, series_df["pred_q10"], series_df["pred_q90"], alpha=0.10, color="tab:blue", label="Neblina P10-P90")
    plt.fill_between(x, series_df["pred_q25"], series_df["pred_q75"], alpha=0.18, color="tab:blue", label="Neblina P25-P75")

    # Linhas brutas (sem suavizacao).
    plt.plot(x, series_df["real_value"], color="black", linewidth=2.0, label="Valor real (nao suavizado)")
    plt.plot(x, series_df["pred_maxmin"], color="tab:red", linewidth=1.8, label="Regra A: maioria+maxmin")
    plt.plot(x, series_df["pred_quantile"], color="tab:orange", linewidth=1.4, alpha=0.9, label="Regra B: quantil")
    plt.plot(x, series_df["pred_conf_gate"], color="tab:purple", linewidth=1.6, alpha=0.9, label="Regra C: gate confianca")
    plt.plot(x, series_df["pred_conf_weighted"], color="tab:green", linewidth=1.8, alpha=0.9, label="Regra D: gate + peso confianca")

    title_suffix = " [invertida]" if INVERT_PREDICTION else ""
    plt.title(f"Comparacao diaria das regras vs real{title_suffix}")
    plt.xlabel("Dia")
    plt.ylabel("Retorno")
    plt.grid(True, alpha=0.25)
    plt.legend(ncol=2, fontsize=9)
    plt.tight_layout()
    plt.savefig(out_path, dpi=140)
    plt.close()


def evaluate_rules(series_df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    y = series_df["real_value"].to_numpy(dtype=float)
    rule_cols = ["pred_maxmin", "pred_quantile", "pred_conf_gate", "pred_conf_weighted"]
    cost_rate = TRANSACTION_COST_BPS / 10_000.0

    pred_rows: list[dict] = []
    strat_rows: list[dict] = []
    strategy_series = series_df[["time_hour", "real_value"]].copy()

    for col in rule_cols:
        p = series_df[col].to_numpy(dtype=float)
        err = p - y

        mae = float(np.mean(np.abs(err)))
        rmse = float(np.sqrt(np.mean(err**2)))
        sign_mask = p != 0.0
        hit_ratio = float(np.mean(np.sign(p[sign_mask]) == np.sign(y[sign_mask]))) if np.any(sign_mask) else np.nan
        corr = float(np.corrcoef(p, y)[0, 1]) if len(p) > 1 else np.nan

        denom = np.abs(y) + np.abs(p) + EPS
        smape = float(np.mean(2.0 * np.abs(err) / denom) * 100.0)
        y_mean = float(np.mean(y))
        ss_res = float(np.sum((y - p) ** 2))
        ss_tot = float(np.sum((y - y_mean) ** 2))
        r2 = float(1.0 - ss_res / (ss_tot + EPS))

        y_std = float(np.std(y))
        rmse_score = float(np.clip(1.0 - rmse / (y_std + EPS), 0.0, 1.0) * 100.0)
        hit_score = float(np.nan_to_num(hit_ratio) * 100.0)
        smape_score = float(np.clip(100.0 - smape, 0.0, 100.0))
        r2_score = float(np.clip((r2 + 1.0) / 2.0, 0.0, 1.0) * 100.0)
        pred_score = float(0.35 * hit_score + 0.30 * rmse_score + 0.20 * smape_score + 0.15 * r2_score)

        pred_rows.append(
            {
                "mode": "invertido" if INVERT_PREDICTION else "normal",
                "rule": col,
                "n_obs": int(len(series_df)),
                "mae": mae,
                "rmse": rmse,
                "smape_pct": smape,
                "r2": r2,
                "hit_ratio_pct_nonzero": float(np.nan_to_num(hit_ratio) * 100.0),
                "corr": corr,
                "pred_score_0_100": pred_score,
            }
        )

        # Backtest t+1 com custo por turnover.
        raw_pos = np.sign(p)
        exec_pos = pd.Series(raw_pos).shift(EXECUTION_LAG_DAYS).fillna(0.0).to_numpy(dtype=float)
        prev_exec_pos = pd.Series(exec_pos).shift(1).fillna(0.0).to_numpy(dtype=float)
        turnover = np.abs(exec_pos - prev_exec_pos)
        ret_gross = exec_pos * y
        ret_net = ret_gross - turnover * cost_rate
        equity = np.cumprod(1.0 + ret_net)

        strategy_series[f"pos_{col}"] = exec_pos
        strategy_series[f"ret_{col}"] = ret_net
        strategy_series[f"equity_{col}"] = equity

        n = len(ret_net)
        total_return = float(equity[-1] - 1.0) if n > 0 else np.nan
        avg_ret = float(np.mean(ret_net)) if n > 0 else np.nan
        std_ret = float(np.std(ret_net, ddof=1)) if n > 1 else np.nan
        sharpe = float((avg_ret / std_ret) * np.sqrt(ANNUALIZATION_DAILY)) if (std_ret and std_ret > 0) else np.nan
        running_max = np.maximum.accumulate(equity) if n > 0 else np.array([np.nan])
        drawdown = equity / running_max - 1.0 if n > 0 else np.array([np.nan])
        max_dd = float(np.min(drawdown)) if n > 0 else np.nan
        avg_turnover = float(np.mean(turnover)) if n > 0 else np.nan
        active_ratio = float(np.mean(np.abs(exec_pos) > 0.0)) if n > 0 else np.nan

        strat_rows.append(
            {
                "mode": "invertido" if INVERT_PREDICTION else "normal",
                "rule": col,
                "execution_lag_days": EXECUTION_LAG_DAYS,
                "transaction_cost_bps": TRANSACTION_COST_BPS,
                "total_return_pct": total_return * 100.0,
                "sharpe_annualized": sharpe,
                "max_drawdown_pct": max_dd * 100.0,
                "avg_turnover": avg_turnover,
                "active_ratio": active_ratio,
            }
        )

    pred_df = pd.DataFrame(pred_rows).sort_values(["pred_score_0_100", "rmse"], ascending=[False, True]).reset_index(drop=True)
    strat_df = pd.DataFrame(strat_rows).sort_values(["total_return_pct", "sharpe_annualized"], ascending=[False, False]).reset_index(drop=True)
    strategy_series = strategy_series.sort_values("time_hour").reset_index(drop=True)
    return pred_df, strat_df, strategy_series


def plot_rule_equity(strategy_series: pd.DataFrame, strat_df: pd.DataFrame, out_path: Path) -> None:
    plt.figure(figsize=(14, 6))
    x = strategy_series["time_hour"]
    for i, rule in enumerate(strat_df["rule"].tolist()):
        col = f"equity_{rule}"
        if col not in strategy_series.columns:
            continue
        lw = 2.0 if i == 0 else 1.4
        alpha = 0.95 if i < 2 else 0.75
        label = f"{rule} (ret={strat_df.loc[strat_df['rule'] == rule, 'total_return_pct'].iloc[0]:.2f}%)"
        plt.plot(x, strategy_series[col], linewidth=lw, alpha=alpha, label=label)
    plt.title("Curva de capital por regra (t+1 dia)")
    plt.xlabel("Dia")
    plt.ylabel("Capital acumulado (base 1.0)")
    plt.grid(True, alpha=0.25)
    plt.legend(fontsize=9)
    plt.tight_layout()
    plt.savefig(out_path, dpi=140)
    plt.close()


def main() -> None:
    df = load_samples(SAMPLES_PATH)
    df = filter_last_n_days(df, PLOT_LAST_N_DAYS)
    if df.empty:
        raise ValueError("Sem dados no recorte de dias solicitado para plot.")

    series_df = build_hourly_series(df)
    plot_real_vs_majority_with_smooth_fog(series_df, OUT_PLOT)

    out = series_df.copy()
    out["time_hour"] = pd.to_datetime(out["time_hour"]).dt.strftime("%Y-%m-%d %H:%M:%S")
    out.to_csv(OUT_SERIES, index=False)
    pred_score_df, strat_df, strategy_series = evaluate_rules(series_df)
    pred_score_df.to_csv(OUT_SCORE, index=False)
    strategy_series_out = strategy_series.copy()
    strategy_series_out["time_hour"] = pd.to_datetime(strategy_series_out["time_hour"]).dt.strftime("%Y-%m-%d %H:%M:%S")
    strategy_series_out.to_csv(OUT_STRATEGY, index=False)
    plot_rule_equity(strategy_series, strat_df, OUT_RULE_PLOT)

    print(f"Plot salvo em: {OUT_PLOT}")
    print(f"Series salvas em: {OUT_SERIES}")
    print(f"Score salvo em: {OUT_SCORE}")
    print(f"Estrategias (serie) salvas em: {OUT_STRATEGY}")
    print(f"Plot de capital por regra salvo em: {OUT_RULE_PLOT}")
    print(
        "Janela plotada (diaria): {} | {} -> {}".format(
            f"{PLOT_LAST_N_DAYS} dias",
            pd.Timestamp(series_df["time_hour"].min()).strftime("%Y-%m-%d %H:%M"),
            pd.Timestamp(series_df["time_hour"].max()).strftime("%Y-%m-%d %H:%M"),
        )
    )
    if not pred_score_df.empty:
        best_pred = pred_score_df.iloc[0]
        print(
            "Melhor previsao | regra={} | hit_nonzero={:.2f}% | RMSE={:.6f} | sMAPE={:.2f}% | R2={:.4f} | score={:.2f}/100".format(
                best_pred["rule"],
                float(best_pred["hit_ratio_pct_nonzero"]),
                float(best_pred["rmse"]),
                float(best_pred["smape_pct"]),
                float(best_pred["r2"]),
                float(best_pred["pred_score_0_100"]),
            )
        )
    if not strat_df.empty:
        best_strat = strat_df.iloc[0]
        print(
            "Melhor estrategia t+1 dia | regra={} | retorno={:.2f}% | Sharpe={:.3f} | MaxDD={:.2f}% | turnover={:.4f}".format(
                best_strat["rule"],
                float(best_strat["total_return_pct"]),
                float(best_strat["sharpe_annualized"]),
                float(best_strat["max_drawdown_pct"]),
                float(best_strat["avg_turnover"]),
            )
        )


if __name__ == "__main__":
    main()
