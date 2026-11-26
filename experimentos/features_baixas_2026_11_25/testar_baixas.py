"""Testa atributos de baixas no pool simplificado de 21 pregões."""

from hashlib import sha256
import os
from pathlib import Path
import sys

import numpy as np
import pandas as pd

OUT = Path(__file__).resolve().parent
PREVIOUS = OUT.parent / 'apos_pausa_features_2026_11_25'
sys.path.insert(0, str(PREVIOUS))
from comparar_cenarios import TICKERS, compute_metrics, legacy, predict, prepare

os.environ['MPLCONFIGDIR'] = str(OUT / 'cache_matplotlib')
GROUPS = {'mercado_em_queda': ['beta_baixas_63', 'fracao_risco_baixa_ibov_21'],
          'fraqueza_recente': ['drawdown_21', 'retorno_relativo_5'],
          'pressao_vendedora': ['volume_em_dias_de_queda_21', 'salto_abertura']}


def indicators(path):
    raw = pd.read_csv(path, parse_dates=['date']).set_index('date')
    returns = np.log(raw.close).diff()
    market = np.log(raw.ibov_close).diff()
    down = market.lt(0)
    market_down = market.where(down).rolling(63, min_periods=15)
    volume = raw.volume.where(raw.volume.gt(0))
    variance = market.pow(2).rolling(21).sum().replace(0, np.nan)
    return pd.DataFrame({
        'beta_baixas_63': market_down.cov(returns.where(down)) / market_down.var().replace(0, np.nan),
        'fracao_risco_baixa_ibov_21': market.clip(upper=0).pow(2).rolling(21).sum() / variance,
        'drawdown_21': raw.close / raw.close.rolling(21).max() - 1,
        'retorno_relativo_5': (returns - market).rolling(5).sum(),
        'volume_em_dias_de_queda_21': volume.where(returns.lt(0), 0).rolling(21).sum() / volume.rolling(21).sum(),
        'salto_abertura': np.log(raw.open.where(raw.open.gt(0)) / raw.close.shift(1)),
    }).replace([np.inf, -np.inf], np.nan)


def main():
    reference = PREVIOUS / 'horizonte_21' / 'previsoes.csv'
    files = {ticker: legacy.DATA_DIR / f'{ticker}.csv' for ticker in TICKERS}
    protected = [reference, *files.values(), *OUT.parents[1].glob('**/*.tex')]
    hashes = {path: sha256(path.read_bytes()).hexdigest() for path in protected}
    frames, sets = prepare(files)
    pool = pd.concat([frames[ticker, 21].join(indicators(path), on='date', validate='one_to_one')
                      for ticker, path in files.items()], ignore_index=True).sort_values(['date', 'ticker'])
    identities = pd.get_dummies(pool.ticker, prefix='ticker', dtype=np.float32)
    pool = pd.concat([pool, identities], axis=1).reset_index(drop=True)
    base = sets['simplificadas'] + identities.columns.tolist()
    individual_features = {column: [column] for group in GROUPS.values() for column in group}
    variants = {'referencia': [], **GROUPS, **individual_features}
    results = []
    for name, extra in variants.items():
        prediction, _ = predict(pool, base + extra, 3)
        results.append(prediction.assign(modelo=name))
        print(f'{name}: previsões concluídas.', flush=True)
    forecasts = pd.concat(results, ignore_index=True)
    keys = ['ticker', 'horizonte', 'date', 'fase', 'fold']
    historical = pd.read_csv(reference, parse_dates=['date']).query("cenario == 'simplificadas' and modelo == 'pool'")
    checked = forecasts.query("modelo == 'referencia'").merge(historical, on=keys, validate='one_to_one')
    assert len(checked) == len(historical) and len(checked)*len(variants) == len(forecasts)
    np.testing.assert_allclose(checked.previsao_x, checked.previsao_y, rtol=0, atol=1e-12)
    assert all(sha256(path.read_bytes()).hexdigest() == digest for path, digest in hashes.items())
    forecasts.to_csv(OUT / 'previsoes.csv', index=False)

    recortes = [('apos_pausa', forecasts[forecasts.dias_corridos.gt(1)]), ('todos', forecasts)]
    metrics = pd.concat([compute_metrics(rows, ['fase', 'ticker', 'modelo']).assign(recorte=name)
                         for name, rows in recortes], ignore_index=True)
    metrics.to_csv(OUT / 'metricas_por_acao.csv', index=False)
    summary = metrics.groupby(['recorte', 'fase', 'modelo']).agg(
        n=('n','sum'), rmse=('rmse','mean'), acerto_altas=('altas_pct','mean'),
        acerto_baixas=('baixas_pct','mean'), balanceada=('balanceada_pct','mean')).reset_index()
    summary.to_csv(OUT / 'resumo.csv', index=False)
    print(summary.to_string(index=False))


if __name__ == '__main__':
    main()
