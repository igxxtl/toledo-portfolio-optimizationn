import hashlib,json,os,shutil,subprocess,sys
from functools import partial
from pathlib import Path

sys.dont_write_bytecode=True
os.environ.setdefault('OPENBLAS_NUM_THREADS','1')
os.environ.setdefault('OMP_NUM_THREADS','1')
import numpy as np
import pandas as pd

repo=Path('../GitHub/toledo-portfolio-optimizationn').resolve()
sys.path.insert(0,str(repo))
from intel.Blacklitterman import simular_pool_vs_atual as module
from intel.Blacklitterman import config as cfg

engine=module.engine
ref=module.OUT
base=Path('experimentos/backtest_base_noticias_2026_11_25').resolve()
version=sys.argv[1]
out=base/version
out.mkdir(parents=True,exist_ok=True)
protocol=json.loads((ref/'protocolo_financeiro.json').read_text(encoding='utf-8'))
assert protocol['arms']==module.ARMS and protocol['rebalance']=='monthly'
assert protocol['seed']==2025 and protocol['window']==126 and protocol['risk']==3 and protocol['cost_bps']==10 and protocol['subsets_per_n']==30
original_manifest=json.loads((cfg.OUTPUTS_DIR/'dois_cenarios/manifest.json').read_text(encoding='utf-8'))
for k,v in original_manifest['bl_parameters'].items():
    assert np.isclose(getattr(cfg,k),v),k

source=out/'entrada_noticias.json'
if version=='anterior':
    blob=subprocess.check_output(['git','-C',str(repo),'show','HEAD:intel/data_acquisition/noticias_b3_sem_duplicados_sentimento.json'])
    assert len(json.loads(blob))==4681
else:
    blob=cfg.NEWS_PATH.read_bytes()
    assert len(json.loads(blob))==5044
source.write_bytes(blob)
shutil.copyfile(ref/'previsoes_pool.csv',out/'previsoes_pool.csv')
module.OUT=out
prices,returns,paths=engine.load_inputs()
loader=engine.load_daily_sentiment_for_ticker
try:
    engine.load_daily_sentiment_for_ticker=partial(loader,news_path=source)
    dates,signals,pred_paths=module.make_signals(prices,returns)
finally:
    engine.load_daily_sentiment_for_ticker=loader
assert len(dates)==250
assert len(signals)==12
old_subsets=pd.read_csv(ref/'subconjuntos.csv')
results,curves,trades=[],[],[]
for n in protocol['n']:
    subsets=engine.subsets(n,len(returns.columns),protocol['subsets_per_n'],protocol['seed'])
    assert [';'.join(returns.columns[j] for j in subset) for subset in subsets]==old_subsets.loc[old_subsets.n.eq(n)].sort_values('repetition').assets.tolist()
    cases=[('mesmos_ativos',i,s) for i,s in enumerate(subsets)]+[('escolha_livre',0,None)]
    for scenario, repetition, subset in cases:
        for arm in module.ARMS:
            metrics,curve,trade=engine.simulate(dates,returns,signals,n,arm,subset,protocol['risk'],protocol['cost_bps'])
            labels=dict(scenario=scenario,n=n,repetition=repetition,model=arm)
            results.append(dict(**labels,**metrics))
            if scenario=='escolha_livre':
                curves.append(curve.assign(**labels))
                trades.append(trade.assign(**labels))
    print(version,n,len(results),flush=True)
result=pd.DataFrame(results)
assert len(result)==3997, len(result)
result.to_csv(out/'metricas.csv',index=False)
pd.concat(curves).to_csv(out/'series_selecao_propria.csv',index=False)
pd.concat(trades).to_csv(out/'rebalanceamentos_selecao_propria.csv',index=False)
hashes={str(p):hashlib.sha256(p.read_bytes()).hexdigest() for p in [*paths,*pred_paths,out/'previsoes_pool.csv',source]}
(out/'execucao.json').write_text(json.dumps(dict(config=protocol,bl_parameters=original_manifest['bl_parameters'],hashes=hashes),ensure_ascii=False,indent=2),encoding='utf-8')

