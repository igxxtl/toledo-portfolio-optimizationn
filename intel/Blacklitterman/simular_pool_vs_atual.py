"""Compara o pool fixo com o pipeline atual, em motor financeiro identico.

python -m intel.Blacklitterman.simular_pool_vs_atual
Nao sobrescreve os resultados historicos nem recalibra parametros em 2025.
"""
from pathlib import Path
import hashlib
import json
import numpy as np
import pandas as pd
from . import simular_dois_cenarios as engine
from . import config as cfg

ROOT=Path(__file__).resolve().parents[2]
OUT=ROOT/'analise_resultados/simulacao_pool'
ARMS=['BL_atual','BL_pool','BL_pool_omega_fixo','XGB_atual','XGB_pool','Prior','Markowitz']


def fixed_omega_posterior(pi,cov,omega,q,tau):
    prior_cov=tau*cov
    return pi+prior_cov@np.linalg.solve(prior_cov+np.diag(omega),q-pi)


def make_signals(prices,returns):
    current,choices,paths=engine.prepare_predictions(prices,pd.Timestamp('2025-01-01'))
    pooled=pd.read_csv(OUT/'previsoes_pool.csv',parse_dates=['date'])
    new={}
    for asset,p in current.items():
        selected=pooled[(pooled.ticker==asset.removesuffix('.SA'))&(pooled.horizon==p['h'])]
        if selected.date.duplicated().any():raise ValueError('Previsoes duplicadas')
        new[asset]=dict(h=p['h'],actual=p['actual'],maturity=p['maturity'],
                        pred=selected.set_index('date').y_pred.reindex(prices.index))
    args=(returns,prices)
    dates,old_signals,old_audit=engine.build_signals(*args,current,pd.Timestamp('2025-01-01'),pd.Timestamp('2025-12-31'),126,3.)
    new_dates,new_signals,new_audit=engine.build_signals(*args,new,pd.Timestamp('2025-01-01'),pd.Timestamp('2025-12-31'),126,3.)
    if not dates.equals(new_dates) or set(old_signals)!=set(new_signals):raise AssertionError('Calendario diferente')
    if len(old_audit)!=240 or len(new_audit)!=240:raise AssertionError('Auditoria mensal incompleta')
    if not (new_audit.prediction_date==new_audit.signal_date).all():raise AssertionError('Previsao desatualizada')
    # Mesmo historico realizado maduro e mesmas noticias: muda somente o previsor.
    for col in ['news_count','horizon','prior','sentiment_component','mature_errors']:
        np.testing.assert_allclose(old_audit[col],new_audit[col],rtol=1e-10,atol=1e-12)
    rf=(1+cfg.SELIC_ANNUAL)**(1/252)-1
    signals={};controlled=[]
    for execution,old in old_signals.items():
        signal=old['signal_date']
        a=old_audit[old_audit.signal_date.eq(signal)].set_index('asset').loc[returns.columns]
        b=new_audit[new_audit.signal_date.eq(signal)].set_index('asset').loc[returns.columns]
        if old['signal_date']!=new_signals[execution]['signal_date']:raise AssertionError('Sinal diferente')
        coverage=np.minimum(1.,a.news_count.to_numpy()/cfg.NEWS_COUNT_MIN)
        quantitative_weight=1-(1-cfg.ALPHA_Q)*coverage
        q_fixed=a.Q.to_numpy()+quantitative_weight*(b.prediction.to_numpy()-a.prediction.to_numpy())/a.horizon.to_numpy()
        np.testing.assert_allclose(q_fixed,b.Q.to_numpy(),atol=1e-12,rtol=0)
        fixed=fixed_omega_posterior(a.prior.to_numpy(),old['cov'],a.omega.to_numpy(),q_fixed,cfg.TAU)
        signals[execution]=dict(signal_date=signal,cov=old['cov'],BL_atual=old['BL'],BL_pool=new_signals[execution]['BL'],
            BL_pool_omega_fixo=fixed,XGB_atual=a.prediction.to_numpy()/a.horizon.to_numpy()-rf,
            XGB_pool=b.prediction.to_numpy()/b.horizon.to_numpy()-rf,Prior=a.prior.to_numpy(),Markowitz=old['Markowitz'])
        for j,asset in enumerate(returns.columns):
            controlled.append(dict(signal_date=signal,execution_date=execution,asset=asset,
                prediction_old=float(a.prediction.iloc[j]),prediction_pool=float(b.prediction.iloc[j]),
                Q_old=float(a.Q.iloc[j]),Q_pool=float(q_fixed[j]),omega_old=float(a.omega.iloc[j]),
                omega_pool=float(b.omega.iloc[j]),posterior_old=float(old['BL'][j]),posterior_pool=float(new_signals[execution]['BL'][j]),
                posterior_fixed_omega=float(fixed[j]),prior=float(a.prior.iloc[j]),news_count=int(a.news_count.iloc[j])))
    old_audit.to_csv(OUT/'auditoria_atual.csv',index=False)
    new_audit.to_csv(OUT/'auditoria_pool.csv',index=False)
    pd.DataFrame(controlled).to_csv(OUT/'auditoria_comparada.csv',index=False)
    choices.to_csv(OUT/'horizontes_mantidos.csv',index=False)
    return dates,signals,paths


def main():
    OUT.mkdir(parents=True,exist_ok=True)
    reference=cfg.OUTPUTS_DIR/'dois_cenarios'
    old_manifest=json.loads((reference/'manifest.json').read_text(encoding='utf-8'))
    for key,value in old_manifest['bl_parameters'].items():
        if not np.isclose(getattr(cfg,key),value):raise ValueError('Parametro alterado: '+key)
    old_files=list(reference.glob('*.csv'))+list(reference.glob('*.json'))
    old_hashes={str(p.relative_to(ROOT)):hashlib.sha256(p.read_bytes()).hexdigest() for p in old_files}
    # Protocolo congelado antes da simulacao; nao existe otimizacao pelos retornos abaixo.
    (OUT/'protocolo_financeiro.json').write_text(json.dumps(dict(arms=ARMS,n=list(range(1,21)),
        seed=2025,subsets_per_n=30,window=126,risk=3,cost_bps=10,rebalance='monthly',
        execution='signal close, trade next trading close, previous holdings accrue returns first',
        confidence='BL_pool recalcula confianca apenas dos erros maduros do novo modelo; controle preserva Omega original.',
        horizons='Mantidos exatamente os escolhidos em 2024 na rodada atual.',
        limitation='Pool destacou-se na avaliacao de 2025; backtest retrospectivo, sem teste independente.',
        reference_hashes=old_hashes),ensure_ascii=False,indent=2),encoding='utf-8')
    prices,returns,source_paths=engine.load_inputs()
    dates,signals,pred_paths=make_signals(prices,returns)
    rows,curves,trades,subsets=[],[],[],[]
    for n in range(1,21):
        common=engine.subsets(n,len(returns.columns),30,2025)
        cases=[('mesmos_ativos',i,s) for i,s in enumerate(common)]+[('escolha_livre',0,None)]
        for scenario,repetition,subset in cases:
            if subset is not None:subsets.append(dict(n=n,repetition=repetition,assets=';'.join(returns.columns[i] for i in subset)))
            for arm in ARMS:
                m,c,t=engine.simulate(dates,returns,signals,n,arm,subset,3.,10.)
                tags=dict(scenario=scenario,n=n,repetition=repetition,model=arm)
                rows.append(dict(**tags,**m))
                if scenario=='escolha_livre':curves.append(c.assign(**tags));trades.append(t.assign(**tags))
        print(f'n={n}: {len(common)} subconjuntos + selecao propria, sete configuracoes concluidas.',flush=True)
    result=pd.DataFrame(rows)
    result.to_csv(OUT/'metricas.csv',index=False)
    pd.concat(curves).to_csv(OUT/'series_selecao_propria.csv',index=False)
    pd.concat(trades).to_csv(OUT/'rebalanceamentos_selecao_propria.csv',index=False)
    pd.DataFrame(subsets).to_csv(OUT/'subconjuntos.csv',index=False)
    # Reproducao numerica dos 1142 resultados financeiros da rodada original.
    historical=pd.read_csv(reference/'metricas.csv')
    current=result[result.model.isin(['BL_atual','Markowitz'])].copy()
    current['model']=current.model.replace({'BL_atual':'BL'})
    check=current.merge(historical,on=['scenario','n','repetition','model'],suffixes=('_new','_old'),validate='one_to_one')
    if len(check)!=len(historical):raise AssertionError('Cobertura da referencia diferente')
    max_errors={}
    for col in ['return_pct','final_capital_brl','sharpe','max_drawdown_pct','turnover_total','cost_brl']:
        error=float(np.max(np.abs(check[col+'_new']-check[col+'_old'])));max_errors[col]=error
        np.testing.assert_allclose(check[col+'_new'],check[col+'_old'],atol=1e-7,rtol=1e-10)
    summary=result.groupby(['scenario','n','model'],as_index=False).agg(
        runs=('return_pct','size'),return_median=('return_pct','median'),
        return_p10=('return_pct',lambda s:s.quantile(.1)),return_p90=('return_pct',lambda s:s.quantile(.9)),
        sharpe_median=('sharpe','median'),drawdown_median=('max_drawdown_pct','median'),
        turnover_median=('turnover_total','median'),cost_median=('cost_brl','median'))
    summary.to_csv(OUT/'resumo.csv',index=False)
    paired=result.pivot(index=['scenario','n','repetition'],columns='model',values='return_pct').reset_index()
    paired['BL_pool_minus_current_pp']=paired.BL_pool-paired.BL_atual
    paired['BL_fixed_minus_current_pp']=paired.BL_pool_omega_fixo-paired.BL_atual
    paired['XGB_pool_minus_current_pp']=paired.XGB_pool-paired.XGB_atual
    paired.to_csv(OUT/'comparacao_pareada.csv',index=False)
    paired.groupby(['scenario','n'],as_index=False).agg(
        pairs=('repetition','size'),median_BL_gain_pp=('BL_pool_minus_current_pp','median'),
        wins_BL_pct=('BL_pool_minus_current_pp',lambda s:float((s>1e-8).mean()*100)),
        ties_BL_pct=('BL_pool_minus_current_pp',lambda s:float(np.isclose(s,0,atol=1e-8,rtol=0).mean()*100)),
        median_XGB_gain_pp=('XGB_pool_minus_current_pp','median')).to_csv(OUT/'comparacao_resumo.csv',index=False)
    for name,h in old_hashes.items():
        if hashlib.sha256((ROOT/name).read_bytes()).hexdigest()!=h:raise AssertionError('Resultado antigo alterado')
    (OUT/'verificacao.json').write_text(json.dumps(dict(status='passed',runs=len(result),original_runs_reproduced=len(check),
        max_errors=max_errors,reference_unchanged=True),indent=2),encoding='utf-8')
    paths=source_paths+pred_paths+[OUT/'previsoes_pool.csv',Path(__file__)]
    (OUT/'manifest_financeiro.json').write_text(json.dumps(dict(status='complete',exploratory=True,n_days=len(dates),
        runs=len(result),arms=ARMS,gate=cfg.NEWS_COUNT_MIN,original_reference_unchanged=True,
        hashes={str(p.relative_to(ROOT)):hashlib.sha256(p.read_bytes()).hexdigest() for p in paths}),indent=2),encoding='utf-8')
    print(result[(result.scenario=='escolha_livre')&result.n.isin([5,10,20])][['n','model','return_pct','sharpe','max_drawdown_pct']].to_string(index=False))


if __name__=='__main__':main()
