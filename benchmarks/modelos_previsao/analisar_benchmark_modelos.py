"""Resume previsões pareadas e empacota trilha de auditoria."""
import argparse
from pathlib import Path
import zipfile

import numpy as np
import pandas as pd
from sklearn.metrics import balanced_accuracy_score, roc_auc_score

MODELS = ("XGBoost projeto", "XGBoost regressao", "XGBoost 3 classes",
          "GRU", "Lag-Llama", "LightGBM", "Ensemble XGB-LGBM",
          "Ridge", "Media historica", "Zero")
KEY = ["horizon", "ticker", "date", "target_date"]


def calculate(wide, groups):
    sections = []
    for key, group in wide.groupby(groups, sort=True):
        key = key if isinstance(key, tuple) else (key,)
        y = group.y.to_numpy()
        pos = y > 0
        denom = ((y - y.mean()) ** 2).sum()
        for model in MODELS:
            p = group[model].to_numpy()
            error = p - y
            vary = np.std(p) > 1e-14
            signs = model != "Zero"
            sections.append(dict(zip(groups, key)) | dict(
                modelo=model, n=len(y), n_datas=group.date.nunique(),
                acuracia_direcional=np.mean(np.sign(p) == np.sign(y)) if signs else np.nan,
                acuracia_balanceada=balanced_accuracy_score(pos, p > 0) if signs and np.unique(pos).size == 2 else np.nan,
                auc=roc_auc_score(pos, p) if signs and np.unique(pos).size == 2 else np.nan,
                correlacao=np.corrcoef(y, p)[0, 1] if vary and denom > 0 else np.nan,
                mae=np.abs(error).mean(), rmse=np.sqrt((error**2).mean()), vies=error.mean(),
                r2=1-(error**2).sum()/denom if denom else np.nan,
                proporcao_positivos=pos.mean()))
    return pd.DataFrame(sections)


def intervals(wide):
    output = []
    rng = np.random.default_rng(42)
    for h, z in wide.groupby("horizon"):
        reference = z["XGBoost projeto"].to_numpy()
        y = z.y.to_numpy()
        for model in ("GRU", "Lag-Llama", "LightGBM", "Ensemble XGB-LGBM", "Ridge", "Media historica", "Zero"):
            pred = z[model].to_numpy()
            daily = pd.DataFrame(dict(
                date=z.date, n=1, mae=np.abs(pred-y)-np.abs(reference-y),
                mse_model=(pred-y)**2, mse_xgb=(reference-y)**2,
                direcao=(np.sign(pred)==np.sign(y)).astype(int)-(np.sign(reference)==np.sign(y)).astype(int),
            )).groupby("date").sum()
            a = daily.to_numpy()
            for length in sorted({max(10, h), max(20, h*2)}):
                starts = rng.integers(len(a), size=(4000, int(np.ceil(len(a)/length))))
                block_idx = ((starts[...,None]+np.arange(length))%len(a)).reshape(4000,-1)[:,:len(a)]
                boot = a[block_idx].sum(axis=1)
                observed = a.sum(axis=0, keepdims=True)
                for arr, name in [(boot, "boot"), (observed, "point")]:
                    metrics = np.column_stack((arr[:,1]/arr[:,0], np.sqrt(arr[:,2]/arr[:,0])-np.sqrt(arr[:,3]/arr[:,0]),
                                               arr[:,4]/arr[:,0]))
                    if name == "boot":
                        bounds = np.percentile(metrics, [2.5,97.5], axis=0)
                    else:
                        for k, metric in enumerate(("mae", "rmse", "acuracia_direcional")):
                            if model != "Zero" or k != 2:
                                output.append(dict(horizon=h, modelo=model, metrica=metric, bloco=length,
                                                   diferenca=metrics[0,k], ic95_inf=bounds[0,k], ic95_sup=bounds[1,k]))
    return pd.DataFrame(output)


def fmt(v, digits=2):
    return "—" if pd.isna(v) else f"{v:.{digits}f}".replace(".", ",")


def plot_results(overall, out):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(1, 2, figsize=(10.6, 4.7), layout="constrained")
    colors = {"XGBoost projeto": "#a24520", "GRU": "#245c99", "Lag-Llama": "#227650",
              "LightGBM": "#a17507", "Ridge": "#805aa7", "Media historica": "#6b6a68", "Zero": "#292929"}
    for name, color in colors.items():
        frame = overall.loc[overall.modelo.eq(name)].sort_values("horizon")
        for ax, field, factor in zip(axes, ("rmse", "acuracia_balanceada"), (100, 100)):
            if frame[field].notna().any():
                ax.plot(frame.horizon, frame[field]*factor, label=name, color=color,
                        marker="o", linestyle="--" if name in {"Media historica", "Zero"} else "-")
    axes[0].set_title("Erro de magnitude")
    axes[0].set_ylabel("RMSE do log-retorno (%)")
    axes[1].set_title("Direção, com peso igual para alta e baixa")
    axes[1].set_ylabel("Acurácia balanceada (%)")
    axes[1].axhline(50, color="#aaa", linewidth=1, linestyle=":")
    for ax in axes:
        ax.set_xticks([1, 5, 10, 21])
        ax.set_xlabel("Horizonte em pregões")
        ax.grid(axis="y", alpha=.25)
    axes[0].legend(frameon=False)
    fig.suptitle("Benchmark temporal dos 20 ativos em 2025", fontsize=14)
    fig.savefig(out/"comparativo_benchmark.png", dpi=175)
    plt.close(fig)


def archived_predictions(wide, repo, path):
    frames = []
    for (h, ticker), _ in wide.loc[wide.horizon.ne(1)].groupby(["horizon", "ticker"]):
        file = repo/"criacao_modelo_xgb"/f"{ticker}_h{h}d_predicoes.csv"
        frame = pd.read_csv(file, parse_dates=["date"], usecols=["date", "y_real", "y_pred"])
        frames.append(frame.assign(horizon=h, ticker=ticker))
    archived = wide.loc[wide.horizon.ne(1), KEY+["y"]].merge(
        pd.concat(frames), on=["horizon", "ticker", "date"], validate="one_to_one", how="left")
    assert np.allclose(archived.y, archived.y_real, atol=1e-13, rtol=0)
    assert np.isfinite(archived.y_pred).all()
    archived = archived.drop(columns="y_real")
    archived.to_csv(path/"previsoes_xgboost_arquivadas_2025.csv", index=False)
    result = []
    for h, z in archived.groupby("horizon"):
        error = z.y_pred-z.y
        result.append(dict(horizon=h, n=len(z), rmse=np.sqrt((error**2).mean()),
                           mae=error.abs().mean(),
                           direcao=(np.sign(z.y_pred)==np.sign(z.y)).mean()))
    return pd.DataFrame(result)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--inputs", type=Path, required=True)
    p.add_argument("--out", type=Path, required=True)
    p.add_argument("--repo", type=Path, required=True)
    args = p.parse_args()
    path = args.inputs
    parts = [pd.read_csv(path / f"previsoes_{tag}.csv", parse_dates=["date", "target_date"])
             for tag in ("tabulares", "gru", "lag", "lightgbm")]
    combined = pd.concat(parts, ignore_index=True)
    test = combined.loc[combined.phase.eq("teste")].copy()
    assert np.isfinite(test[["pred", "y"]]).all().all()
    assert not test.duplicated(KEY+["model"]).any()
    assert test.date.dt.year.eq(2025).all() and test.target_date.dt.year.eq(2025).all()
    actual = test.loc[test.model.eq("XGBoost projeto"), KEY+["y"]].rename(columns={"y": "realizado"})
    test = test.merge(actual, on=KEY, validate="many_to_one", how="left")
    assert np.allclose(test.y, test.realizado, atol=1e-13, rtol=0)
    test["y"] = test.pop("realizado")
    ensemble = test.loc[test.model.eq("XGBoost regressao")].merge(
        test.loc[test.model.eq("LightGBM"), KEY+["pred"]].rename(columns={"pred":"lgbm"}),
        on=KEY, validate="one_to_one")
    ensemble["pred"] = .5 * (ensemble.pred + ensemble.pop("lgbm"))
    ensemble["model"] = "Ensemble XGB-LGBM"
    test = pd.concat([test, ensemble], ignore_index=True)
    counts = test.groupby(["horizon","model"]).size().unstack()
    wide = test.pivot(index=KEY+["y"], columns="model", values="pred").reset_index()
    if wide[list(MODELS)].isna().any().any():
        raise ValueError(f"Unpaired predictions: {counts}")
    assert wide.ticker.nunique() == 20
    wide["trimestre"] = wide.date.dt.quarter
    overall = calculate(wide, ["horizon"])
    asset = calculate(wide, ["horizon","ticker"])
    quarter = calculate(wide, ["horizon","trimestre"])
    interval = intervals(wide)
    archived = archived_predictions(wide, args.repo, path)
    archived.to_csv(path/"metricas_xgboost_arquivado.csv", index=False)
    plot_results(overall, args.out)
    asset.to_csv(path/"metricas_por_ativo.csv", index=False)
    quarter.to_csv(path/"metricas_por_trimestre.csv", index=False)
    overall.to_csv(path/"metricas_agregadas.csv", index=False)
    interval.to_csv(path/"intervalos_vs_xgboost.csv", index=False)
    test.to_csv(path/"previsoes_pareadas_2025.csv", index=False)
    parts[0].loc[parts[0].phase.eq("validacao")].to_csv(path/"validacao_xgboost_2024.csv", index=False)
    with zipfile.ZipFile(args.out/"resultados_benchmark_modelos.zip", "w", zipfile.ZIP_DEFLATED) as z:
        for name in ("metricas_por_ativo.csv","metricas_por_trimestre.csv","metricas_agregadas.csv",
                     "intervalos_vs_xgboost.csv","previsoes_pareadas_2025.csv","validacao_xgboost_2024.csv",
                     "selecao_tabular.json","lag_checkpoint.json","ambiente.txt",
                     "metricas_xgboost_arquivado.csv","previsoes_xgboost_arquivadas_2025.csv"):
            z.write(path/name, name)
    out = []
    for h, t in overall.groupby("horizon"):
        d = wide.loc[wide.horizon.eq(h)]
        positive_rate = fmt((d.y > 0).mean()*100)
        n = f"{len(d):,}".replace(",", ".")
        horizon_label = "1 pregão" if h == 1 else f"{h} pregões"
        out.append(f"### {horizon_label}\n\n{n} pares ativo-data em {d.date.nunique()} datas, com origens de {d.date.min():%d/%m/%Y} a {d.date.max():%d/%m/%Y}. A proporção de retornos positivos foi {positive_rate}%.\n")
        out.append("| Modelo | RMSE (%) | MAE (%) | Direção (%) | Bal. (%) | AUC | R² |\n|---|---:|---:|---:|---:|---:|---:|")
        for row in t.sort_values("rmse").itertuples():
            out.append(f"| {row.modelo} | {fmt(row.rmse*100,3)} | {fmt(row.mae*100,3)} | {fmt(row.acuracia_direcional*100)} | {fmt(row.acuracia_balanceada*100)} | {fmt(row.auc,3)} | {fmt(row.r2,3)} |")
        out.append("")
    out.append("### Diferenças pareadas contra XGBoost projeto\n")
    out.append("| Pregões | Modelo | Δ RMSE (p.p.) e IC 95% | Δ MAE (p.p.) e IC 95% | Δ direção (p.p.) e IC 95% |\n|---:|---|---|---|---|")
    subset = interval.loc[interval.bloco.eq(interval.horizon.clip(lower=10))]
    for (h, model), t in subset.groupby(["horizon","modelo"]):
        fields = t.set_index("metrica")
        values = []
        for metric in ("rmse","mae","acuracia_direcional"):
            if metric not in fields.index:
                values.append("—")
            else:
                row = fields.loc[metric]
                values.append(f"{fmt(row.diferenca*100,3)} [{fmt(row.ic95_inf*100,3)} a {fmt(row.ic95_sup*100,3)}]")
        out.append("| "+f"{h} | {model} | "+" | ".join(values)+" |")
    out.extend(["", "### Diagnóstico adicional do XGBoost arquivado no projeto", "",
                "O recorte usa os CSVs originais e valida data, ativo, horizonte e igualdade do retorno realizado. A trilha arquivada usa o protocolo original e fica fora do ranking e dos intervalos do novo benchmark.", "",
                "| Pregões | Observações | RMSE (%) | MAE (%) | Direção (%) |",
                "|---:|---:|---:|---:|---:|"])
    for a in archived.itertuples():
        out.append(f"| {a.horizon} | {a.n} | {fmt(a.rmse*100,3)} | {fmt(a.mae*100,3)} | {fmt(a.direcao*100)} |")
    (path/"tabelas.md").write_text("\n".join(out), encoding="utf-8")
    print(overall[["horizon", "modelo", "n", "rmse", "mae", "acuracia_direcional", "auc"]].round(4).to_string(index=False))


if __name__ == "__main__":
    main()
