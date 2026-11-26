"""Compara atributos atuais e econômicos com o mesmo treinamento temporal."""

from hashlib import sha256
import json
import os
from pathlib import Path
import sys

import numpy as np
import pandas as pd
import xgboost

OUT = Path(__file__).resolve().parent
THESIS = OUT.parents[1]
REPO = THESIS.parent / 'GitHub' / 'toledo-portfolio-optimizationn'
SOURCE = REPO / 'analise_resultados' / 'simulacao_pool'
os.environ['MPLCONFIGDIR'] = str(OUT / 'cache_matplotlib')
sys.path.insert(0, str(REPO))

from intel.model_training import train_xgb as legacy
from intel.model_training.experimento_features import ANCHOR, augmented
from intel.model_training.experimento_melhoria import FOLDS, TICKERS, fit_predict, masks, metric
from intel.model_training.experimento_metodologia import fit_rows
from intel.model_training.features_experimento import breadth_frames, select_columns


def economic_features(raw):
    price, index = raw.close, raw.ibov_close
    returns = np.log(price).diff()
    market = np.log(index).diff()
    beta = returns.rolling(63).cov(market) / market.rolling(63).var().replace(0, np.nan)
    residual = returns - beta.shift(1) * market
    traded = (price * raw.volume).where(raw.volume > 0)
    illiquidity = (price.pct_change().abs() / traded).rolling(21, min_periods=18).mean()
    return pd.DataFrame({
        'eco_beta_63': beta,
        'eco_momentum_252_21': np.log(price.shift(21) / price.shift(252)),
        'eco_relativo_ibov_21': (returns - market).rolling(21).sum(),
        'eco_residual_21': residual.rolling(21).sum(),
        'eco_vol_residual_21': residual.rolling(21).std(),
        'eco_vol_quedas_21': np.sqrt(returns.clip(upper=0).pow(2).rolling(21).mean()),
        'eco_amihud_21': np.log1p(illiquidity * 1e6),
        'eco_momentum_ibov_21': market.rolling(21).sum(),
    }, index=raw.index).replace([np.inf, -np.inf], np.nan)


def main():
    protocol_path = REPO / 'analise_resultados/metodologia_xgb/protocolo.json'
    groups = json.loads(protocol_path.read_text(encoding='utf-8'))['groups_selected_only_in_2024']
    inputs = [legacy.DATA_DIR / f'{ticker}.csv' for ticker in TICKERS]
    original = SOURCE / 'previsoes_pool.csv'
    protected = [*THESIS.glob('**/*.tex'), *inputs, *legacy.OUT_DIR.glob('*_predicoes.csv'), original,
                 protocol_path, SOURCE / 'metricas.csv', Path(legacy.__file__)]
    before = {path: sha256(path.read_bytes()).hexdigest() for path in protected}
    saved = pd.read_csv(original, parse_dates=['date'])
    prices = pd.DataFrame({ticker: pd.read_csv(path, parse_dates=['date']).set_index('date').close
                           for ticker, path in zip(TICKERS, inputs)})
    breadth, ranks = breadth_frames(prices), legacy.build_cross_sectional_ranks(inputs)
    factors = {ticker: economic_features(pd.read_csv(path, parse_dates=['date']).set_index('date'))
               for ticker, path in zip(TICKERS, inputs)}
    forecasts, audits = [], []
    for horizon in (5, 10, 21):
        frames = []
        for ticker in TICKERS:
            frame, pools = augmented(ticker, horizon, ranks, breadth, legacy)
            observed = factors[ticker].reindex(frame.date).reset_index(drop=True)
            frames.append(pd.concat([frame.assign(ticker=ticker, horizon=horizon), observed], axis=1))
        pool = pd.concat(frames, ignore_index=True).sort_values(['date', 'ticker']).reset_index(drop=True)
        columns = pools[groups[str(horizon)]]
        economics = observed.columns.tolist()
        identities = [f'identity_{ticker}' for ticker in TICKERS]
        for ticker, identity in zip(TICKERS, identities):
            pool[identity] = pool.ticker.eq(ticker).astype(float)
        for fold, start, end, phase in FOLDS:
            train, test = masks(pool, start, end, phase)
            # Também exporta dezembro para a simulação, com a mesma regra do projeto.
            export = np.flatnonzero((pool.date.ge(start) & pool.date.lt(end) & pool.current.notna()).to_numpy())
            assert pool.maturity.iloc[train].max() < pd.Timestamp(start)
            block = pool.iloc[export][['ticker', 'horizon', 'date', 'maturity', 'actual']].copy()
            block['avaliar'] = np.isin(export, test)
            block['phase'], block['fold'] = phase, fold
            y = pool.actual.to_numpy(float)
            for name, candidates in [('atual_retreinado', columns), ('economico', economics),
                                     ('atual_mais_economico', columns + economics)]:
                selected, _ = select_columns(pool, fit_rows(pool, train), candidates + identities)
                pred, audit = fit_predict(pool[selected].to_numpy(np.float32), y, pool, train, export,
                                          ANCHOR, 400, 2)
                block[name] = pred
                audits.append(dict(horizon=horizon, fold=fold, arm=name,
                                   features=';'.join(c for c in selected if c not in identities), **audit))
            forecasts.append(block)
            print(f'{horizon} pregões, {fold}: três conjuntos treinados', flush=True)
    predictions = pd.concat(forecasts, ignore_index=True).merge(
        saved[['ticker', 'horizon', 'date', 'y_pred']].rename(columns={'y_pred': 'atual_publicado'}),
        on=['ticker', 'horizon', 'date'], validate='one_to_one')
    assert len(predictions) == sum(map(len, forecasts)) == len(saved)
    predictions.to_csv(OUT / 'previsoes.csv', index=False)
    pd.DataFrame(audits).to_csv(OUT / 'auditoria.csv', index=False)
    arms = ['atual_retreinado', 'economico', 'atual_mais_economico', 'atual_publicado', 'zero']
    predictions['zero'] = 0.
    rows = []
    for (phase, horizon, ticker), data in predictions[predictions.avaliar].groupby(['phase', 'horizon', 'ticker']):
        for arm in arms:
            rows.append(dict(phase=phase, horizon=horizon, ticker=ticker, arm=arm,
                             **metric(data.actual, data[arm])))
    scores = pd.DataFrame(rows)
    scores.to_csv(OUT / 'metricas_por_acao.csv', index=False)
    summary = scores.groupby(['phase', 'horizon', 'arm'])[['rmse', 'mae', 'hr', 'skill_zero']].mean()
    summary.to_csv(OUT / 'resumo.csv')
    assert all(sha256(path.read_bytes()).hexdigest() == digest for path, digest in before.items())
    (OUT / 'verificacao.json').write_text(json.dumps(dict(verified_files=len(before), rows=len(predictions),
        xgboost=xgboost.__version__, pandas=pd.__version__, numpy=np.__version__,
        max_abs_diff_baseline=float((predictions.atual_retreinado-predictions.atual_publicado).abs().max()),
        horizon_groups=groups, sample='retrospectiva 2025, sem seleção de atributos pelo teste'), indent=2), encoding='utf-8')
    print(summary.to_string(), flush=True)


if __name__ == '__main__':
    main()
