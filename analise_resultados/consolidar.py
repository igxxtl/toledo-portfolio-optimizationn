"""Consolida os resultados materializados; nao treina ou reotimiza modelos."""
from pathlib import Path
import json
import hashlib
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / 'intel/Blacklitterman/outputs/dois_cenarios'
OUT = ROOT / 'analise_resultados/dados'


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    summary = pd.read_csv(SOURCE / 'resumo.csv')
    metrics = pd.read_csv(SOURCE / 'metricas.csv')
    paired = pd.read_csv(SOURCE / 'comparacao_pareada.csv')
    audit = pd.read_csv(SOURCE / 'auditoria_sinais.csv')
    manifest = json.loads((SOURCE / 'manifest.json').read_text(encoding='utf-8'))
    gate = int(manifest['bl_parameters']['NEWS_COUNT_MIN'])
    choices = pd.read_csv(SOURCE / 'horizontes_validacao.csv')
    series = pd.read_csv(SOURCE / 'series.csv', parse_dates=['date'])
    trades = pd.read_csv(SOURCE / 'rebalanceamentos.csv', parse_dates=['execution_date'])
    forecast_rows = []
    for choice in choices.itertuples():
        ticker = choice.asset.removesuffix('.SA')
        price = pd.read_csv(ROOT / f'dados_diarios/{ticker}.csv', usecols=['date', 'close'])
        price['date'] = pd.to_datetime(price.date)
        price = price.set_index('date').close.sort_index()
        p = pd.read_csv(ROOT / f'criacao_modelo_xgb/{choice.prediction_file}', usecols=['date', 'y_pred'])
        p['date'] = pd.to_datetime(p.date)
        pred = p.set_index('date').y_pred.reindex(price.index)
        real = np.log(price.shift(-choice.horizon) / price)
        mature = pd.Series(price.index, index=price.index).shift(-choice.horizon)
        valid = pred.notna() & real.notna() & (price.index >= '2025-01-01') & mature.le(pd.Timestamp('2025-12-30'))
        y, yp = real[valid], pred[valid]
        baseline = max(float((y > 0).mean()), float((y < 0).mean())) * 100
        r2 = 1 - float(((yp - y)**2).sum() / ((y - y.mean())**2).sum())
        forecast_rows.append(dict(ticker=ticker, horizon=choice.horizon, n_test=len(y),
                                  validation_hr=choice.validation_direction_accuracy * 100,
                                  test_hr=float((np.sign(yp) == np.sign(y)).mean() * 100),
                                  majority_hr=baseline, r2_test=r2,
                                  rmse_test=float(np.sqrt(((yp - y)**2).mean())),
                                  mae_test=float((yp - y).abs().mean())))
    forecasts = pd.DataFrame(forecast_rows)
    forecasts.to_csv(OUT / 'previsao_horizontes_2025.csv', index=False)
    old = pd.read_csv(ROOT / 'criacao_modelo_xgb/metricas_vencedores.csv')
    common = summary[summary.scenario == 'mesmos_ativos'].pivot(index='n', columns='model')
    rows = []
    for n, group in paired[paired.scenario == 'mesmos_ativos'].groupby('n'):
        rows.append(dict(n=n, repetitions=len(group),
                         return_bl=common.loc[n, ('return_median', 'BL')],
                         return_mk=common.loc[n, ('return_median', 'Markowitz')],
                         paired_delta_median=group.BL_minus_Markowitz_pp.median(),
                         win_pct=(group.BL_minus_Markowitz_pp > 1e-8).mean() * 100,
                         tie_pct=(group.BL_minus_Markowitz_pp.abs() <= 1e-8).mean() * 100,
                         bl_beats_selic_pct=(group.BL > 14.872510217327983).mean() * 100,
                         mk_beats_selic_pct=(group.Markowitz > 14.872510217327983).mean() * 100))
    common_comparison = pd.DataFrame(rows)
    common_comparison.to_csv(OUT / 'comparacao_pareada_resumo.csv', index=False)
    # Mesmo caminho de pesos sem o desconto de custos; nao e nova otimizacao.
    keys = ['scenario', 'n', 'repetition', 'model']
    cost = trades[keys + ['execution_date', 'cost_fraction']].rename(columns={'execution_date': 'date'})
    joined = series.merge(cost, on=keys + ['date'], how='left', validate='many_to_one')
    joined['cost_fraction'] = joined.cost_fraction.fillna(0.)
    joined['gross_factor'] = (1 + joined.ret) / (1 - joined.cost_fraction)
    gross = joined.groupby(keys).gross_factor.prod().rename('gross_factor').reset_index()
    gross = metrics.merge(gross, on=keys, validate='one_to_one')
    gross['gross_return_pct'] = (gross.gross_factor - 1) * 100
    gross['cost_impact_pp'] = gross.gross_return_pct - gross.return_pct
    gross.to_csv(OUT / 'custos_contrafactual.csv', index=False)
    cost_summary = gross.groupby(['scenario', 'n', 'model'], as_index=False).agg(
        net_return=('return_pct', 'median'), gross_return=('gross_return_pct', 'median'),
        impact_pp=('cost_impact_pp', 'median'), explicit_cost_brl=('cost_brl', 'median'),
        turnover=('turnover_total', 'median'))
    cost_summary.to_csv(OUT / 'custos_resumo.csv', index=False)
    free = series[series.scenario == 'escolha_livre'].copy()
    monthly = free.assign(month=free.date.dt.to_period('M').astype(str)).groupby(
        ['n', 'model', 'month']).ret.apply(lambda s: ((1 + s).prod() - 1) * 100).reset_index()
    monthly.to_csv(OUT / 'retornos_mensais_livres.csv', index=False)
    coverage = audit.groupby('signal_date').agg(eligible=('news_count', lambda s: int((s >= gate).sum())),
                                             with_news=('news_count', lambda s: int((s > 0).sum())))
    coverage.to_csv(OUT / 'cobertura_noticias_rebalanceamentos.csv')
    selection = trades[trades.scenario == 'escolha_livre'].copy()
    selection.to_csv(OUT / 'composicoes_livres.csv', index=False)
    stats = dict(
        forecasts=dict(horizon_counts=forecasts.horizon.value_counts().to_dict(),
                       mean_validation_hr=forecasts.validation_hr.mean(), mean_test_hr=forecasts.test_hr.mean(),
                       test_hr_min=forecasts.test_hr.min(), test_hr_max=forecasts.test_hr.max(),
                       negative_test_r2=int((forecasts.r2_test < 0).sum()),
                       below_majority=int((forecasts.test_hr < forecasts.majority_hr).sum()),
                       above_majority=int((forecasts.test_hr > forecasts.majority_hr).sum()),
                       original_mean_hr=old.dir_acc_pct.mean(), original_negative_r2=int((old.r2_wf < 0).sum()),
                       original_horizon_counts=old.horizon.value_counts().to_dict()),
        news=dict(gate=gate, observations=len(audit), with_news=int((audit.news_count > 0).sum()),
                  eligible=int((audit.news_count >= gate).sum()), dates_with_any_eligible=int((coverage.eligible > 0).sum())),
        n20=metrics[(metrics.n == 20) & (metrics.scenario == 'escolha_livre')].to_dict('records'),
        source_hashes={str(p.relative_to(ROOT)): hashlib.sha256(p.read_bytes()).hexdigest()
                       for p in [SOURCE/'manifest.json', SOURCE/'metricas.csv', SOURCE/'resumo.csv', SOURCE/'series.csv',
                                 ROOT/'criacao_modelo_xgb/metricas_vencedores.csv']})
    old_h = old.set_index('ticker').horizon
    new_h = choices.assign(ticker=choices.asset.str.replace('.SA', '', regex=False)).set_index('ticker').horizon
    stats['forecasts']['changed_horizons'] = int((new_h.reindex(old_h.index) != old_h).sum())
    (OUT / 'estatisticas.json').write_text(json.dumps(stats, ensure_ascii=False, indent=2), encoding='utf-8')
    print(json.dumps(stats, ensure_ascii=True, indent=2))
    print(common_comparison.to_string(index=False))
    print(forecasts.to_string(index=False))
    print(cost_summary[(cost_summary.scenario == 'escolha_livre') & cost_summary.n.isin([1,2,5,10,20])].to_string(index=False))


if __name__ == '__main__':
    main()
