"""Diagnosticos com baselines observaveis e resultados maturados ate fim de 2025."""
import numpy as np
import pandas as pd
from .camadas_intermediarias import OUT, ROOT
from intel.Blacklitterman import simular_dois_cenarios as engine


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    prices, _, _ = engine.load_inputs()
    selected, _, _ = engine.prepare_predictions(prices, pd.Timestamp('2025-01-01'))
    rows, observations, features = [], [], []
    for asset, p in selected.items():
        h = p['h']
        # Media de alvos ja realizados: shift(h) impede consulta ao futuro.
        historical = p['actual'].shift(h).expanding(min_periods=80).mean()
        mask = ((prices.index.year == 2025) & p['maturity'].le(pd.Timestamp('2025-12-30'))
                & p['pred'].notna() & p['actual'].notna() & historical.notna())
        frame = pd.DataFrame(dict(actual=p['actual'][mask], xgb=p['pred'][mask],
                                  zero=0., historical=historical[mask]))
        observations.append(frame.reset_index(names='date').assign(asset=asset, horizon=h))
        groups = [('2025', frame)] + [(str(q), g) for q, g in frame.groupby(frame.index.to_period('Q'))]
        for period, g in groups:
            for model in ('xgb', 'zero', 'historical'):
                err = g[model] - g.actual
                rows.append(dict(asset=asset, horizon=h, period=period, model=model, n=len(g),
                                 rmse=float(np.sqrt((err**2).mean())), mae=float(err.abs().mean()),
                                 directional_accuracy=float((np.sign(g[model]) == np.sign(g.actual)).mean())))
        path = ROOT / f'criacao_modelo_xgb/{asset.removesuffix(".SA")}_h{h}d_shap.csv'
        if path.exists():
            features.append(pd.read_csv(path).assign(asset=asset, horizon=h,
                            scope='Artefato historico; nao interpretado como SHAP exclusivo de 2025'))
    pd.DataFrame(rows).to_csv(OUT / 'xgb_baselines_trimestres.csv', index=False)
    pd.concat(observations).to_csv(OUT / 'xgb_previsto_realizado.csv', index=False)
    pd.concat(features).to_csv(OUT / 'shap_historico_horizontes_selecionados.csv', index=False)


if __name__ == '__main__':
    main()
