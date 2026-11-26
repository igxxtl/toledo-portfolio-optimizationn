import json
from pathlib import Path
import numpy as np
import pandas as pd

base=Path('experimentos/backtest_base_noticias_2026_11_25')
ref=Path('../GitHub/toledo-portfolio-optimizationn/analise_resultados/simulacao_pool')
old,new=(base/name for name in ('anterior','nova'))
keys=['scenario','n','repetition','model']
historical=pd.read_csv(ref/'metricas.csv').set_index(keys).sort_index()
reexecuted=(old/'metricas.csv').is_file()
earlier=pd.read_csv(old/'metricas.csv').set_index(keys).sort_index() if reexecuted else historical
current=pd.read_csv(new/'metricas.csv').set_index(keys).sort_index()
assert not earlier.index.has_duplicates
assert len(historical)==len(earlier)==len(current)==3997
assert historical.index.equals(earlier.index) and earlier.index.equals(current.index)
np.testing.assert_allclose(earlier,historical,atol=1e-7,rtol=1e-10)
controls=['XGB_atual','XGB_pool','Prior','Markowitz']
selection=earlier.index.get_level_values('model').isin(controls)
np.testing.assert_allclose(earlier.loc[selection],current.loc[selection],atol=1e-7,rtol=1e-10)
manifests=[json.loads((p/'execucao.json').read_text(encoding='utf-8')) for p in ((old,new) if reexecuted else (new,new))]
assert manifests[0]['config']==manifests[1]['config']
assert manifests[0]['bl_parameters']==manifests[1]['bl_parameters']
hashes=[{('previsoes_pool.csv' if k.endswith('previsoes_pool.csv') else k):v
         for k,v in manifest['hashes'].items() if not k.endswith('entrada_noticias.json')}
        for manifest in manifests]
assert hashes[0]==hashes[1]
if not reexecuted:
    historical_manifest=json.loads((ref/'manifest_financeiro.json').read_text(encoding='utf-8'))
    root=ref.resolve().parents[1]
    relevant={str((root/relative).resolve()):digest for relative,digest in historical_manifest['hashes'].items()
              if relative.startswith(('dados_diarios','criacao_modelo_xgb'))}
    assert relevant and all(hashes[1].get(filename)==digest for filename,digest in relevant.items())
    previous_memory=historical_manifest['hashes'][r'intel\data_acquisition\noticias_b3_sem_duplicados_sentimento.json']
    from hashlib import sha256
    news=(old/'entrada_noticias.json').read_bytes()
    expected_hashes={sha256(data).hexdigest() for data in (news, news.replace(b'\r\n',b'\n'),news.replace(b'\r\n',b'\n').replace(b'\n',b'\r\n'))}
    assert previous_memory in expected_hashes
fields=['return_pct','sharpe','max_drawdown_pct','volatility_pct','turnover_total','cost_brl','final_capital_brl']
paired=earlier[fields].add_suffix('_anterior').join(current[fields].add_suffix('_nova'),validate='one_to_one')
for field in fields:
    paired['delta_'+field]=paired[field+'_nova']-paired[field+'_anterior']
paired=paired.reset_index()
paired.to_csv(base/'comparacao_pareada.csv',index=False)
comparisons=[]
for (scenario,n,model),group in paired.groupby(['scenario','n','model']):
    row=dict(scenario=scenario,n=n,model=model,pares=len(group))
    for field in fields:
        for suffix in ('anterior','nova'):
            row[field+'_'+suffix]=group[field+'_'+suffix].median()
        row['mediana_delta_'+field]=group['delta_'+field].median()
    change=group.delta_return_pct
    row.update(vitorias=int((change>1e-8).sum()),empates=int((change.abs()<=1e-8).sum()),derrotas=int((change<-1e-8).sum()))
    comparisons.append(row)
summary=pd.DataFrame(comparisons)
summary.to_csv(base/'resumo_por_carteira.csv',index=False)
key=['signal_date','execution_date','news_date','asset']
audit_old=pd.read_csv(old/'auditoria_pool.csv').set_index(key).sort_index()
audit_new=pd.read_csv(new/'auditoria_pool.csv').set_index(key).sort_index()
audit_ref=pd.read_csv(ref/'auditoria_pool.csv').set_index(key).sort_index()
assert audit_old.index.equals(audit_new.index) and len(audit_old)==240
pd.testing.assert_frame_equal(audit_old,audit_ref,check_exact=False,atol=1e-10,rtol=1e-9)
pd.testing.assert_frame_equal(pd.read_csv(old/'horizontes_mantidos.csv'),pd.read_csv(new/'horizontes_mantidos.csv'))
for field in ['prediction_date','horizon','prediction','mature_errors','max_outcome_date','prior']:
    pd.testing.assert_series_equal(audit_old[field],audit_new[field])
audit_old.add_suffix('_anterior').join(audit_new.add_suffix('_nova')).to_csv(base/'auditoria_comparada.csv')
coverage=pd.DataFrame([dict(base=name,pares=len(a),com_noticia=int(a.news_count.gt(0).sum()),
                            gate_ativo=int(a.gate_active.sum()),noticias_total=int(a.news_count.sum()))
                       for name,a in [('anterior',audit_old),('nova',audit_new)]])
coverage.to_csv(base/'cobertura.csv',index=False)
check=dict(resultados_base_anterior=len(earlier),base_anterior_reexecutada=reexecuted,execucao_nova=len(current),controles_sem_variacao=int(selection.sum()),
           erro_absoluto_maximo_reproducao=float((earlier-historical).abs().to_numpy().max()) if reexecuted else None,
           gate_minimo=manifests[0]['bl_parameters'].get('NEWS_COUNT_MIN'))
(base/'verificacao.json').write_text(json.dumps(check,indent=2),encoding='utf-8')
print(check,flush=True)
print(coverage.to_string(index=False),flush=True)
models=['BL_atual','BL_pool','BL_pool_omega_fixo']
cols=['scenario','n','model','pares','return_pct_anterior','return_pct_nova','mediana_delta_return_pct','sharpe_anterior','sharpe_nova','max_drawdown_pct_anterior','max_drawdown_pct_nova','vitorias','empates','derrotas']
print(summary[summary.model.isin(models)&summary.n.isin([5,10,20])][cols].round(4).to_string(index=False))
for m in models:
    for scenario in ['mesmos_ativos','escolha_livre']:
        vals=paired.query('model==@m and scenario==@scenario').delta_return_pct
        print(m,scenario,len(vals),float(vals.median()),int((vals>1e-8).sum()),int((vals<-1e-8).sum()))



