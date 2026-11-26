"""Calibra em 2024 uma interpolação da mediana até o quantil direcional."""

from hashlib import sha256
import json
import os
from pathlib import Path
import sys

import numpy as np
import pandas as pd

OUT = Path(__file__).resolve().parent
PARENT = OUT.parent
sys.path.insert(0, str(PARENT))
from comparar import FOLDS, TICKERS, fit_quantiles, indicators, legacy, prepare

os.environ['MPLCONFIGDIR'] = str(OUT / 'cache_matplotlib')


def interpolate(frame, fraction):
    limit = np.select([frame.p50.gt(0),frame.p50.lt(0)], [frame.p90,frame.p10], default=0.)
    return frame.p50 + fraction*(limit-frame.p50)


def metrics(frame, prediction):
    g = frame.assign(erro=frame.actual-prediction).groupby('ticker')
    mse = g.erro.apply(lambda s:s.pow(2).mean())
    variance = g.actual.var(ddof=0)
    assert len(g)==20 and variance.gt(0).all()
    return dict(mae=100*g.erro.apply(lambda s:s.abs().mean()).mean(),rmse=100*np.sqrt(mse).mean(),
                r2=(1-mse/variance).mean())


def main():
    previous = PARENT / 'previsoes_pareadas_2025.csv'
    files = {t:legacy.DATA_DIR/f'{t}.csv' for t in TICKERS}
    protected = [previous,*files.values(),*legacy.OUT_DIR.glob('*_predicoes.csv'),
                 *OUT.parents[2].glob('**/*.tex')]
    hashes = {p:sha256(p.read_bytes()).hexdigest() for p in protected}
    frames,sets = prepare(files)
    pool = pd.concat([frames[t,21].join(indicators(path)[['beta_baixas_63']],on='date',validate='one_to_one')
                     for t,path in files.items()],ignore_index=True).sort_values(['date','ticker'])
    ids = pd.get_dummies(pool.ticker,prefix='ticker',dtype=np.float32)
    pool = pd.concat([pool,ids],axis=1).reset_index(drop=True)
    x = pool[sets['simplificadas']+list(ids)+['beta_baixas_63']].to_numpy(np.float32)
    y = pool.actual.to_numpy(float)
    blocks = []
    for fold in FOLDS:
        if fold[-1]=='validation':
            blocks.append(fit_quantiles(pool,x,y,fold)[0])
            print(f'{fold[0]}: quantis de validação reconstruídos.',flush=True)
    val = pd.concat(blocks,ignore_index=True).query('dias_corridos > 1')
    assert len(val)==980 and val.maturity.lt(pd.Timestamp('2025-01-01')).all()
    grid = pd.DataFrame([dict(proporcao=i/100,**metrics(val,interpolate(val,i/100))) for i in range(101)])
    selected = grid.sort_values(['mae','proporcao']).iloc[0]
    fraction = float(selected.proporcao)
    test = pd.read_csv(previous)
    assert len(test)==1020 and test.dias_corridos.gt(1).all()
    assert (test.p10.le(test.p50)&test.p50.le(test.p90)).all()
    test['previsao_interpolada'] = interpolate(test,fraction)
    assert np.array_equal(np.sign(test.p50),np.sign(test.previsao_interpolada))
    rows = [('pool_atual',test.previsao),('p50',test.p50),('interpolada',test.previsao_interpolada),
            ('ate_limite',interpolate(test,1.))]
    result = pd.DataFrame([dict(modelo=name,n=len(test),**metrics(test,pred)) for name,pred in rows])
    assert all(sha256(p.read_bytes()).hexdigest()==digest for p,digest in hashes.items())
    grid.to_csv(OUT/'grade_2024.csv',index=False)
    result.to_csv(OUT/'resumo_2025.csv',index=False)
    test.to_csv(OUT/'previsoes_2025.csv',index=False)
    (OUT/'protocolo.json').write_text(json.dumps(dict(escolha_2024=selected.to_dict(),
        observacoes_2024=len(val),observacoes_2025=len(test),arquivos_conferidos=len(hashes)),indent=2),encoding='utf-8')
    print(grid.sort_values(['mae','proporcao']).head().to_string(index=False),flush=True)
    print(result.to_string(index=False))


if __name__ == '__main__':
    main()
