"""Calibra em 2024 a influência do ranking pontual na previsão quantílica."""

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
from comparar import FOLDS, KEYS, TICKERS, fit, fit_quantiles, indicators, legacy, prepare

os.environ['MPLCONFIGDIR'] = str(OUT / 'cache_matplotlib')


def add_rank(frame):
    assert not frame.duplicated(['date','ticker']).any()
    group = frame.groupby('date').previsao
    count = group.transform('size')
    assert count.le(20).all()
    rank = group.rank(method='average')
    score = (2*(rank-1)/(count-1).replace(0,np.nan)-1).fillna(0.)
    limit = np.where(score.ge(0),frame.p90,frame.p10)
    return frame.assign(ranking=rank, deslocamento=np.abs(score)*(limit-frame.p50))


def metrics(frame, predicted):
    g = frame.assign(erro=frame.actual-predicted).groupby('ticker')
    mse = g.erro.apply(lambda e:e.pow(2).mean())
    var = g.actual.var(ddof=0)
    assert len(g)==20 and var.gt(0).all()
    return dict(mae=100*g.erro.apply(lambda e:e.abs().mean()).mean(),
                rmse=100*np.sqrt(mse).mean(), r2=(1-mse/var).mean())


def main():
    source = PARENT/'previsoes_pareadas_2025.csv'
    files = {t:legacy.DATA_DIR/f'{t}.csv' for t in TICKERS}
    protected = [source,*files.values(),*legacy.OUT_DIR.glob('*_predicoes.csv'),
                 *OUT.parents[2].glob('**/*.tex')]
    hashes = {p:sha256(p.read_bytes()).hexdigest() for p in protected}
    frames, sets = prepare(files)
    data = pd.concat([frames[t,21].join(indicators(path)[['beta_baixas_63']], on='date', validate='one_to_one')
                      for t,path in files.items()],ignore_index=True).sort_values(['date','ticker'])
    ids = pd.get_dummies(data.ticker,prefix='ticker',dtype=np.float32)
    data = pd.concat([data,ids],axis=1).reset_index(drop=True)
    x = data[sets['simplificadas']+list(ids)+['beta_baixas_63']].to_numpy(np.float32)
    y = data.actual.to_numpy(float)
    parts = []
    for fold in FOLDS:
        if fold[-1]!='validation':
            continue
        quantiles = fit_quantiles(data,x,y,fold)[0]
        direct = fit(data,x,y,1.04,fold)[0]
        merged = quantiles.merge(direct[KEYS+['actual','previsao']],on=KEYS,validate='one_to_one')
        assert len(merged)==len(quantiles)==len(direct)
        np.testing.assert_allclose(merged.actual_x,merged.actual_y,atol=1e-12,rtol=0)
        parts.append(merged.rename(columns={'actual_x':'actual'}).drop(columns='actual_y'))
        print(f'{fold[0]}: previsões de validação reconstruídas.',flush=True)
    valid = add_rank(pd.concat(parts,ignore_index=True)).query('dias_corridos > 1')
    assert len(valid)==980 and valid.maturity.lt(pd.Timestamp('2025-01-01')).all()
    grid = pd.DataFrame([dict(proporcao=i/100,**metrics(valid,valid.p50+(i/100)*valid.deslocamento))
                         for i in range(101)])
    choice = grid.sort_values(['mae','proporcao']).iloc[0]
    fraction = float(choice.proporcao)
    test = add_rank(pd.read_csv(source))
    assert len(test)==1020 and test.dias_corridos.gt(1).all()
    test['previsao_calibrada'] = test.p50+fraction*test.deslocamento
    assert test.previsao_calibrada.ge(test.p10-1e-12).all() and test.previsao_calibrada.le(test.p90+1e-12).all()
    models = {'pool_atual':test.previsao,'p50':test.p50,'ranking_calibrado':test.previsao_calibrada,
              'ranking_integral':test.p50+test.deslocamento}
    result = pd.DataFrame([dict(modelo=name,n=len(test),**metrics(test,pred)) for name,pred in models.items()])
    assert all(sha256(p.read_bytes()).hexdigest()==digest for p,digest in hashes.items())
    grid.to_csv(OUT/'grade_2024.csv',index=False)
    result.to_csv(OUT/'resumo_2025.csv',index=False)
    test.to_csv(OUT/'previsoes_2025.csv',index=False)
    (OUT/'protocolo.json').write_text(json.dumps(dict(escolha_2024=choice.to_dict(),amostra_2024=len(valid),
        amostra_2025=len(test),arquivos_conferidos=len(hashes)),indent=2),encoding='utf-8')
    print(grid.sort_values(['mae','proporcao']).head().to_string(index=False),flush=True)
    print(result.to_string(index=False),flush=True)


if __name__ == '__main__':
    main()
