"""Ajusta peso de retornos negativos em 2024 e testa a escolha congelada em 2025."""

from hashlib import sha256
import json
import os
from pathlib import Path
import sys

import numpy as np
import pandas as pd
from xgboost import XGBRegressor

OUT = Path(__file__).resolve().parent
PARENT = OUT.parent
sys.path.insert(0, str(PARENT))
sys.path.insert(0, str(PARENT / 'calibracao_limiar'))
from calibrar import calibrate, evaluate
from testar_baixas import TICKERS, indicators, legacy, prepare
from comparar_modelos import FOLDS

os.environ['MPLCONFIGDIR'] = str(OUT / 'cache_matplotlib')
WEIGHTS = (1., 1.005, 1.01, 1.02, 1.04, 1.075, 1.10, 1.20, 1.35, 1.50, 2.)
KEYS = ['ticker','date','maturity','horizonte','fase','fold']


def weights(y, multiplier):
    values = np.where(y < 0, multiplier, 1.)
    return values / values.mean()


def fit(frame, x, y, multiplier, fold):
    label, start, end, phase = fold
    start, end = pd.Timestamp(start), pd.Timestamp(end)
    cutoff = pd.Timestamp('2025-01-01' if phase == 'validation' else '2025-12-31')
    train = np.flatnonzero((frame.date.lt(start) & frame.maturity.lt(start)).to_numpy())
    test = np.flatnonzero((frame.date.ge(start) & frame.date.lt(end) & frame.maturity.lt(cutoff)).to_numpy())
    cut = int(len(train)*.8)
    internal = train[cut:]
    boundary = frame.date.iloc[internal[0]]
    initial = train[:cut]
    initial = initial[frame.maturity.iloc[initial].lt(boundary).to_numpy()]
    assert frame.maturity.iloc[train].max() < start and frame.maturity.iloc[initial].max() < boundary
    params = dict(objective='reg:squarederror', eval_metric='rmse', tree_method='hist',
        learning_rate=.03, subsample=.8, colsample_bytree=.8, max_bin=64, max_depth=4,
        min_child_weight=10, reg_alpha=.1, reg_lambda=1., n_jobs=3, random_state=42, verbosity=0)
    center, scale = float(y[initial].mean()), max(float(y[initial].std()), 1e-8)
    stopping = XGBRegressor(**params, n_estimators=400, early_stopping_rounds=20)
    stopping.fit(x[initial], (y[initial]-center)/scale, sample_weight=weights(y[initial], multiplier),
                 eval_set=[(x[internal], (y[internal]-center)/scale)], verbose=False)
    rounds = int(stopping.best_iteration) + 1
    center, scale = float(y[train].mean()), max(float(y[train].std()), 1e-8)
    model = XGBRegressor(**params, n_estimators=rounds)
    model.fit(x[train], (y[train]-center)/scale, sample_weight=weights(y[train], multiplier), verbose=False)
    block = frame.iloc[test][['ticker','date','maturity','horizonte','actual','dias_corridos']].copy()
    block['previsao'] = model.predict(x[test]).astype(float)*scale+center
    block['fase'], block['fold'] = phase, label
    return block, dict(fase=phase, fold=label, peso=multiplier, n_train=len(train),
                       n_baixas_treino=int((y[train]<0).sum()), rounds=rounds,
                       max_train_maturity=str(frame.maturity.iloc[train].max().date()))


def summarize(data, threshold):
    details = evaluate(data, threshold)
    assert details.ticker.nunique() == 20
    columns = ['acerto_altas','acerto_baixas','acuracia_balanceada']
    return dict(n=len(data), limiar=threshold, **details[columns].mean().to_dict())


def main():
    saved_path = PARENT / 'previsoes.csv'
    old_calibration = PARENT / 'calibracao_limiar' / 'protocolo.json'
    files = {ticker: legacy.DATA_DIR / f'{ticker}.csv' for ticker in TICKERS}
    protected = [saved_path, old_calibration, *files.values(), *legacy.OUT_DIR.glob('*_predicoes.csv'),
                 *OUT.parents[2].glob('**/*.tex')]
    hashes = {path: sha256(path.read_bytes()).hexdigest() for path in protected}
    frames, sets = prepare(files)
    pool = pd.concat([frames[ticker,21].join(indicators(path)[['beta_baixas_63']], on='date', validate='one_to_one')
                      for ticker,path in files.items()], ignore_index=True).sort_values(['date','ticker'])
    identities = pd.get_dummies(pool.ticker, prefix='ticker', dtype=np.float32)
    pool = pd.concat([pool,identities], axis=1).reset_index(drop=True)
    x = pool[sets['simplificadas'] + identities.columns.tolist() + ['beta_baixas_63']].to_numpy(np.float32)
    y = pool.actual.to_numpy(float)
    saved = pd.read_csv(saved_path, parse_dates=['date','maturity'])
    saved = saved[saved.dias_corridos.gt(1) & saved.modelo.isin(['referencia','beta_baixas_63'])]
    reference = saved[saved.modelo.eq('referencia')]
    validation_base = reference[reference.fase.eq('validation')]
    target = summarize(validation_base, 0.)['acerto_altas']
    previous_threshold = json.loads(old_calibration.read_text())['limiar_log_retorno']
    audit, candidates = [], []
    for multiplier in WEIGHTS:
        parts = []
        for fold in FOLDS:
            if fold[-1] == 'validation':
                block, log = fit(pool, x, y, multiplier, fold)
                parts.append(block)
                audit.append(log)
        validation = pd.concat(parts, ignore_index=True).query('dias_corridos > 1')
        match = validation.merge(validation_base[KEYS + ['actual']], on=KEYS, validate='one_to_one')
        assert len(match) == len(validation) == len(validation_base)
        np.testing.assert_allclose(match.actual_x, match.actual_y, atol=1e-12, rtol=0)
        if multiplier == 1.:
            control = saved.query("fase == 'validation' and modelo == 'beta_baixas_63'")
            check = validation.merge(control, on=KEYS, validate='one_to_one')
            assert len(check) == len(validation)
            np.testing.assert_allclose(check.previsao_x, check.previsao_y, atol=1e-12, rtol=0)
        threshold, _ = calibrate(validation, target)
        for rule, cutoff in [('zero',0.),('calibrado',threshold)]:
            metrics = summarize(validation, cutoff)
            candidates.append(dict(peso=multiplier, regra=rule, atende_meta=metrics['acerto_altas']>=target-1e-10,
                                   **metrics))
        print(f'Peso {multiplier:g}: validação concluída.', flush=True)

    grid = pd.DataFrame(candidates)
    grid.to_csv(OUT / 'grade_2024.csv', index=False)
    eligible = grid[grid.atende_meta & grid.peso.gt(1)].sort_values(
        ['acerto_baixas','acerto_altas','peso'], ascending=[False,False,True]).iloc[0]
    multiplier, threshold = float(eligible.peso), float(eligible.limiar)
    print(f'Escolha congelada em 2024: peso {multiplier:g}, limiar {threshold:.8f}.', flush=True)
    test_blocks = []
    for fold in FOLDS:
        if fold[-1] == 'test':
            block, log = fit(pool, x, y, multiplier, fold)
            test_blocks.append(block)
            audit.append(log)
    test = pd.concat(test_blocks, ignore_index=True).query('dias_corridos > 1')
    baseline = reference[reference.fase.eq('test')]
    beta = saved.query("fase == 'test' and modelo == 'beta_baixas_63'")
    match = test.merge(baseline[KEYS + ['actual']], on=KEYS, validate='one_to_one')
    assert len(match) == len(test) == len(baseline) == len(beta)
    np.testing.assert_allclose(match.actual_x, match.actual_y, atol=1e-12, rtol=0)
    scenarios = [('referencia',baseline,0.), ('beta_sem_peso',beta,0.),
                 ('beta_sem_peso_calibrado',beta,previous_threshold), ('beta_ponderado_zero',test,0.),
                 ('beta_ponderado_calibrado',test,threshold)]
    summary = pd.DataFrame([dict(variante=name, **summarize(data, cutoff)) for name,data,cutoff in scenarios])
    summary.to_csv(OUT/'resumo_2025.csv', index=False)
    pd.concat([evaluate(data, cutoff).assign(variante=name) for name,data,cutoff in scenarios],
              ignore_index=True).to_csv(OUT/'metricas_por_acao_2025.csv', index=False)
    test.to_csv(OUT/'previsoes_ponderadas_2025.csv', index=False)
    pd.DataFrame(audit).to_csv(OUT/'auditoria.csv', index=False)
    assert all(sha256(path.read_bytes()).hexdigest()==digest for path,digest in hashes.items())
    (OUT/'protocolo.json').write_text(json.dumps(dict(peso_escolhido=multiplier, limiar_escolhido=threshold,
        meta_acerto_altas_2024=target, escolha_2024=eligible.to_dict(), arquivos_conferidos=len(hashes)),
        indent=2),encoding='utf-8')
    print(summary.to_string(index=False))


if __name__ == '__main__':
    main()
