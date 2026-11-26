"""Teste pareado de dois atributos de calendário, somente na previsão."""

from hashlib import sha256
import os
from pathlib import Path
import sys

import numpy as np
import pandas as pd

OUT = Path(__file__).resolve().parent
PREVIOUS = OUT.parent / 'features_economicas_2026_11_25'
os.environ['MPLCONFIGDIR'] = str(OUT / 'cache_matplotlib')
sys.path.insert(0, str(PREVIOUS))

from testar_features import (
    ANCHOR, FOLDS, TICKERS, augmented, breadth_frames, economic_features,
    fit_predict, fit_rows, legacy, masks, metric, select_columns,
)

ARMS = {'pausa': ['dias_corridos'], 'salto': ['gap_abertura'],
        'pausa_e_salto': ['dias_corridos', 'gap_abertura']}


def main():
    reference = PREVIOUS / 'previsoes.csv'
    files = {ticker: legacy.DATA_DIR / f'{ticker}.csv' for ticker in TICKERS}
    protected = [reference, *files.values(), *legacy.OUT_DIR.glob('*_predicoes.csv'),
                 *OUT.parents[1].glob('**/*.tex')]
    hashes = {path: sha256(path.read_bytes()).hexdigest() for path in protected}
    saved = pd.read_csv(reference, parse_dates=['date', 'maturity'])
    raw = {ticker: pd.read_csv(path, parse_dates=['date']).set_index('date')
           for ticker, path in files.items()}
    breadth = breadth_frames(pd.DataFrame({ticker: data.close for ticker, data in raw.items()}))
    ranks = legacy.build_cross_sectional_ranks(list(files.values()))
    factors = {}
    for ticker, data in raw.items():
        assert data.index.is_monotonic_increasing and not data.index.has_duplicates
        factors[ticker] = economic_features(data).assign(
            dias_corridos=data.index.to_series().diff().dt.days,
            gap_abertura=np.log(data.open.where(data.open.gt(0)) / data.close.shift(1)),
        )

    forecasts = []
    for h in (5, 10, 21):
        blocks = []
        for ticker in TICKERS:
            frame, _ = augmented(ticker, h, ranks, breadth, legacy)
            derived = factors[ticker].reindex(frame.date).reset_index(drop=True)
            blocks.append(pd.concat([frame.assign(ticker=ticker, horizon=h), derived], axis=1))
        pool = pd.concat(blocks, ignore_index=True).sort_values(['date', 'ticker']).reset_index(drop=True)
        identities = [f'identity_{ticker}' for ticker in TICKERS]
        for ticker, col in zip(TICKERS, identities):
            pool[col] = pool.ticker.eq(ticker).astype(float)
        economic = [col for col in derived if col.startswith('eco_')]
        y = pool.actual.to_numpy(float)
        for fold, start, end, phase in FOLDS:
            train, test = masks(pool, start, end, phase)
            fit = fit_rows(pool, train)
            assert pool.maturity.iloc[train].max() < pd.Timestamp(start)
            selected, _ = select_columns(pool, fit, economic + identities)
            block = pool.iloc[test][['ticker', 'horizon', 'date', 'maturity', 'actual', 'dias_corridos']].copy()
            for name, extra in {'enxuto': [], **ARMS}.items():
                x = pool[selected + extra].to_numpy(np.float32)
                block[name], _ = fit_predict(x, y, pool, train, test, ANCHOR, 400, 2)
            block['fase'], block['fold'] = phase, fold
            forecasts.append(block)
            print(f'{h} pregões, {fold}: quatro modelos avaliados', flush=True)

    result = pd.concat(forecasts, ignore_index=True)
    check = result.merge(saved[saved.avaliar], on=['ticker', 'horizon', 'date', 'maturity', 'fold'],
                         validate='one_to_one')
    assert len(result) == len(check) == int(saved.avaliar.sum())
    np.testing.assert_allclose(check.enxuto, check.economico, atol=1e-12, rtol=0)
    np.testing.assert_allclose(check.actual_x, check.actual_y, atol=1e-12, rtol=0)
    assert all(sha256(path.read_bytes()).hexdigest() == digest for path, digest in hashes.items())
    result.to_csv(OUT / 'previsoes.csv', index=False)
    metrics = []
    for label, sample in [('todos', result), ('apos_pausa', result[result.dias_corridos.gt(1)])]:
        for (phase, h, ticker), group in sample.groupby(['fase', 'horizon', 'ticker']):
            for model in ('enxuto', *ARMS):
                metrics.append(dict(recorte=label, fase=phase, horizonte=h, ticker=ticker,
                                    modelo=model, **metric(group.actual, group[model])))
    by_asset = pd.DataFrame(metrics)
    by_asset.to_csv(OUT / 'metricas_por_acao.csv', index=False)
    summary = by_asset.groupby(['recorte', 'fase', 'horizonte', 'modelo']).agg(
        observacoes=('n', 'sum'), rmse=('rmse', 'mean'), mae=('mae', 'mean'),
        acerto_direcional=('hr', 'mean'), desempenho_frente_a_zero=('skill_zero', 'mean'))
    summary.to_csv(OUT / 'resumo.csv')
    print(f'{len(hashes)} arquivos originais conferidos; referência reproduzida em {len(check)} previsões.')
    print(summary.to_string())


if __name__ == '__main__':
    main()
