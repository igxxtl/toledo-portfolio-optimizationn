"""Compara R2 do modelo atual e do pool nas mesmas previsoes de teste de 2025."""
from pathlib import Path
import hashlib
import json
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / 'analise_resultados/r2_pool'


def score(frame, model):
    y = frame.actual.to_numpy(float)
    pred = frame[model].to_numpy(float)
    assert len(y) > 1 and np.isfinite(y).all() and np.isfinite(pred).all()
    sst = float(np.sum((y-y.mean())**2))
    assert sst > 0
    sse = float(np.sum((y-pred)**2))
    return dict(n=len(y), r2=1-sse/sst, sse=sse, sst=sst,
                rmse=float(np.sqrt(sse/len(y))))


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    source = ROOT / 'analise_resultados/metodologia_xgb/previsoes.csv'
    data = pd.read_csv(source, parse_dates=['date', 'maturity'])
    data = data[data.phase.eq('test')].copy()
    assert data.date.dt.year.eq(2025).all()
    assert (data.maturity < pd.Timestamp('2025-12-31')).all()
    assert not data.duplicated(['ticker', 'horizon', 'date']).any()
    rows = []
    global_rows = []
    for (ticker, h), frame in data.groupby(['ticker', 'horizon']):
        for model in ['current', 'pool']:
            rows.append(dict(ticker=ticker, horizon=h, model=model, **score(frame, model)))
    by_asset = pd.DataFrame(rows)
    by_asset.to_csv(OUT/'r2_por_ativo.csv', index=False)
    summary = by_asset.groupby(['horizon', 'model'], as_index=False).agg(
        assets=('ticker', 'size'), n=('n', 'sum'), r2_mean=('r2', 'mean'),
        r2_median=('r2', 'median'), positive_r2=('r2', lambda x: int((x>0).sum())),
        rmse_mean=('rmse', 'mean'), sse=('sse', 'sum'), sst_within_assets=('sst', 'sum'))
    summary['r2_within_assets'] = 1-summary.sse/summary.sst_within_assets
    summary.to_csv(OUT/'r2_resumo.csv', index=False)
    for h, frame in data.groupby('horizon'):
        for model in ['current', 'pool']:
            global_rows.append(dict(horizon=h, model=model, **score(frame, model)))
    pd.DataFrame(global_rows).to_csv(OUT/'r2_global.csv', index=False)
    pair = by_asset.pivot(index=['ticker','horizon'], columns='model', values='r2').reset_index()
    pair['improvement'] = pair.pool-pair.current
    pair.to_csv(OUT/'r2_comparacao_por_ativo.csv', index=False)
    # Confere R2 contra as metricas ja calculadas, sem retreinar ou mudar previsoes.
    previous = pd.read_csv(ROOT/'analise_resultados/metodologia_xgb/metricas.csv')
    previous = previous[previous.phase.eq('test') & previous.model.isin(['current','pool'])]
    check = by_asset.merge(previous, on=['ticker','horizon','model'], suffixes=('_calc','_saved'), validate='one_to_one')
    assert len(check) == 120
    np.testing.assert_allclose(check.r2_calc, check.r2_saved, atol=1e-12, rtol=0)
    baseline = pd.read_csv(ROOT/'analise_resultados/evolucao_xgb/modelo_conjunto_exploratorio.csv')
    for model, col in [('current','rmse_current'), ('pool','rmse_pool')]:
        selected = summary[summary.model.eq(model)].set_index('horizon').loc[baseline.horizon]
        np.testing.assert_allclose(selected.rmse_mean, baseline[col], atol=1e-12, rtol=0)
    choices = pd.read_csv(ROOT/'analise_resultados/simulacao_pool/horizontes_mantidos.csv')
    choices['ticker'] = choices.asset.str.replace('.SA','',regex=False)
    selected = pair.merge(choices[['ticker','horizon']], on=['ticker','horizon'], validate='one_to_one')
    assert len(selected) == 20
    selected.to_csv(OUT/'r2_horizontes_da_simulacao.csv', index=False)
    (OUT/'metodologia.txt').write_text(
        'R2 = 1 - soma((realizado-previsao)^2) / soma((realizado-media_realizada)^2).\n'
        'Mesmas observacoes de teste de 2025, mesmos retornos logaritmicos e pool fixo usado na simulacao.\n'
        'Resumo principal: media simples dos R2 calculados por ativo, comparavel a agregacao anterior de MAE/RMSE.\n'
        'r2_global.csv: R2 depois de reunir observacoes de todas as acoes, calculo distinto da media por ativo.\n'
        'r2_within_assets: 1 - soma SSE / soma SST por ativo; pondera os R2 pelas SST.\n'
        'R2 negativo indica erro quadratico maior que o da media realizada da respectiva amostra. '
        'Essa media e uma referencia retrospectiva de avaliacao, nao uma previsao disponivel na origem.\n'
        'R2 nao mede acerto de direcao. Nenhum modelo foi retreinado e a avaliacao continua retrospectiva.\n', encoding='utf-8')
    (OUT/'verificacao.json').write_text(json.dumps(dict(status='passed', paired_asset_horizons=len(pair),
        existing_metrics_matched=len(check), source_sha256=hashlib.sha256(source.read_bytes()).hexdigest(),
        sample='2025 test, same rows for both models'),indent=2), encoding='utf-8')
    print(summary[['horizon','model','assets','n','r2_mean','r2_median','positive_r2','r2_within_assets']].to_string(index=False))
    print('\nR2 global:')
    print(pd.DataFrame(global_rows)[['horizon','model','r2']].to_string(index=False))
    print('\nAtivos que melhoraram por horizonte:')
    print(pair.groupby('horizon').improvement.agg(lambda x: int((x>0).sum())).to_string())


if __name__ == '__main__':
    main()
