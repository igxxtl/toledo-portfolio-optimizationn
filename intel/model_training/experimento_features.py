"""Auditoria e teste isolado de conjuntos de features; sem sobrescrever o pipeline."""
from __future__ import annotations
import argparse
import hashlib
import json
import os
import time
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor,as_completed
import numpy as np
import pandas as pd
from .experimento_melhoria import ROOT,TICKERS,FOLDS,CONFIGS,prepare,masks,fit_predict,metric
from .features_experimento import CORE,breadth_frames,extra_features,select_columns

OUT=ROOT/'analise_resultados/features_xgb'
PREVIOUS=ROOT/'analise_resultados/melhoria_xgb'
GROUPS=['completo','enxuto','enxuto_novo','mercado_ampliado']
ANCHOR=next(c for c in CONFIGS if c['name']=='sq_regular_w0')


def data_audit(legacy,out):
    rows=[];paths=[];closes={};issues=[]
    for name in TICKERS+['IBOV','USDBRL']:
        p=legacy.DATA_DIR/f'{name}.csv';d=pd.read_csv(p,parse_dates=['date']);paths.append(p)
        invalid_price=int((~np.isfinite(d.close)|(d.close<=0)).sum())
        row=dict(asset=name,rows=len(d),start=str(d.date.min().date()),end=str(d.date.max().date()),
            duplicates=int(d.date.duplicated().sum()),sorted=bool(d.date.is_monotonic_increasing),invalid_close=invalid_price)
        if set(['open','high','low']).issubset(d):
            violations=((d.high+1e-6<d[['open','close']].max(axis=1))|
                (d.low-1e-6>d[['open','close']].min(axis=1))|(d.high<d.low))
            row['ohlc_violations']=int(violations.sum())
            row['nonpositive_ohl']=int((d[['open','high','low']]<=0).any(axis=1).sum())
            row['missing_ohlc']=int(d[['open','high','low','close']].isna().any(axis=1).sum())
            for r in d[violations].itertuples():
                issues.append(dict(asset=name,date=str(r.date.date()),open=r.open,high=r.high,low=r.low,close=r.close,
                    within_validation_test=bool(pd.Timestamp('2024-01-01')<=r.date<pd.Timestamp('2026-01-01'))))
        if 'volume' in d:row['nonpositive_volume']=int((d.volume<=0).sum())
        rows.append(row)
        if row['duplicates'] or not row['sorted'] or invalid_price:raise ValueError(f'Dados invalidos: {name}')
        if name in TICKERS:closes[name]=d.set_index('date').close
    audit=pd.DataFrame(rows);audit.to_csv(out/'auditoria_precos.csv',index=False)
    pd.DataFrame(issues).to_csv(out/'inconsistencias_ohlc.csv',index=False)
    return pd.DataFrame(closes).sort_index(),paths


def augmented(ticker,h,ranks,breadth,legacy):
    frame,original=prepare(ticker,h,ranks,legacy)
    raw=legacy.carregar_df(legacy.DATA_DIR/f'{ticker}.csv')
    extra,new,market=extra_features(raw,breadth)
    extra.index=pd.DatetimeIndex(raw.date)
    aligned=extra.reindex(pd.DatetimeIndex(frame.date)).reset_index(drop=True)
    frame=pd.concat([frame,aligned],axis=1)
    core=[c for c in CORE if c in original]
    pools=dict(completo=original,enxuto=core,enxuto_novo=core+new,mercado_ampliado=core+new+market)
    return frame,pools


def audit_original_features(frame,columns,ticker):
    # Diagnostico de redundancia apenas no historico anterior a validacao.
    sample=frame.loc[frame.date<pd.Timestamp('2024-01-01'),columns]
    corr=sample.corr(method='spearman').abs();records=[]
    for i,c in enumerate(columns):
        for j in range(i):
            other=columns[j];rho=corr.iloc[i,j]
            if rho>.98:
                pair=sample[[c,other]].dropna()
                records.append(dict(ticker=ticker,feature=c,other=other,spearman=float(rho),
                    duplicate_numeric=bool(np.allclose(pair[c],pair[other],rtol=1e-9,atol=1e-12))))
    return records


def summarize(out):
    observations=pd.concat([pd.read_csv(p) for p in sorted((out/'cases').glob('*.csv'))],ignore_index=True)
    observations.to_csv(out/'previsoes.csv',index=False)
    rows=[]
    models=GROUPS+['current','previous','zero','historical']
    for (ticker,h,phase),g in observations.groupby(['ticker','horizon','phase']):
        for model in models:rows.append(dict(ticker=ticker,horizon=h,phase=phase,model=model,**metric(g.actual,g[model])))
    metrics=pd.DataFrame(rows);metrics.to_csv(out/'metricas.csv',index=False)
    choices=[];comparison=[]
    for (ticker,h),v in metrics[metrics.phase.eq('validation')].groupby(['ticker','horizon']):
        best=v[v.model.isin(GROUPS)].sort_values(['rmse','model']).iloc[0]
        choices.append(dict(ticker=ticker,horizon=h,chosen=best.model,rmse_validation=best.rmse))
        t=metrics[(metrics.ticker==ticker)&(metrics.horizon==h)&metrics.phase.eq('test')].set_index('model')
        r=t.loc[best.model];current=t.loc['current'];anchor=t.loc['completo'];previous=t.loc['previous']
        comparison.append(dict(ticker=ticker,horizon=h,chosen=best.model,n=int(r.n),
            rmse_current=current.rmse,rmse_new=r.rmse,rmse_anchor=anchor.rmse,rmse_previous=previous.rmse,
            rmse_reduction_pct=(1-r.rmse/current.rmse)*100,feature_gain_pct=(1-r.rmse/anchor.rmse)*100,
            gain_vs_previous_pct=(1-r.rmse/previous.rmse)*100,mae_current=current.mae,mae_new=r.mae,
            hr_current=current.hr,hr_new=r.hr,r2_current=current.r2,r2_new=r.r2,
            skill_current=current.skill_zero,skill_new=r.skill_zero,
            rmse_zero=t.loc['zero'].rmse,rmse_historical=t.loc['historical'].rmse))
    choices=pd.DataFrame(choices);comp=pd.DataFrame(comparison)
    choices.to_csv(out/'escolhas_2024.csv',index=False);comp.to_csv(out/'comparacao_2025.csv',index=False)
    summary=comp.groupby('horizon',as_index=False).agg(assets=('ticker','size'),n=('n','sum'),
        rmse_current=('rmse_current','mean'),rmse_anchor=('rmse_anchor','mean'),rmse_previous=('rmse_previous','mean'),rmse_new=('rmse_new','mean'),
        mae_current=('mae_current','mean'),mae_new=('mae_new','mean'),hr_current=('hr_current','mean'),hr_new=('hr_new','mean'),
        skill_new=('skill_new','mean'),wins_current=('rmse_reduction_pct',lambda s:int((s>0).sum())),
        wins_features=('feature_gain_pct',lambda s:int((s>1e-9).sum())))
    summary['rmse_reduction_pct']=(1-summary.rmse_new/summary.rmse_current)*100
    summary.to_csv(out/'resumo_horizontes.csv',index=False)
    fixed=metrics[metrics.phase.eq('test')].groupby(['horizon','model'],as_index=False).agg(
        assets=('ticker','size'),rmse=('rmse','mean'),mae=('mae','mean'),hr=('hr','mean'),skill_zero=('skill_zero','mean'))
    fixed.to_csv(out/'conjuntos_fixos_2025.csv',index=False)
    old=pd.read_csv(ROOT/'intel/Blacklitterman/outputs/dois_cenarios/horizontes_validacao.csv')
    old['ticker']=old.asset.str.replace('.SA','',regex=False)
    selected=comp.merge(old[['ticker','horizon']],on=['ticker','horizon'],validate='one_to_one')
    selected.to_csv(out/'comparacao_horizontes_atuais.csv',index=False)
    result=dict(cases=len(comp),chosen_counts=choices.chosen.value_counts().to_dict(),
        wins_current=int((comp.rmse_new<comp.rmse_current).sum()),
        wins_anchor=int((comp.feature_gain_pct>1e-9).sum()),wins_previous=int((comp.rmse_new<comp.rmse_previous).sum()),
        wins_zero=int((comp.rmse_new<comp.rmse_zero).sum()),wins_historical=int((comp.rmse_new<comp.rmse_historical).sum()),
        current_horizons=dict(assets=len(selected),rmse_current=float(selected.rmse_current.mean()),
            rmse_previous=float(selected.rmse_previous.mean()),rmse_new=float(selected.rmse_new.mean()),
            hr_current=float(selected.hr_current.mean()),hr_new=float(selected.hr_new.mean())))
    (out/'resultado.json').write_text(json.dumps(result,ensure_ascii=False,indent=2),encoding='utf-8')
    print(summary.to_string(index=False),flush=True);print(json.dumps(result,ensure_ascii=False,indent=2),flush=True)


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output',type=Path,default=OUT)
    parser.add_argument('--tickers',nargs='+',default=TICKERS)
    parser.add_argument('--horizons',nargs='+',type=int,default=[5,10,21])
    parser.add_argument('--workers',type=int,default=4)
    parser.add_argument('--audit-only',action='store_true')
    parser.add_argument('--resume',action='store_true')
    args=parser.parse_args();out=args.output
    out.mkdir(parents=True,exist_ok=True);(out/'cases').mkdir(exist_ok=True)
    os.environ.setdefault('MPLCONFIGDIR',str(ROOT/'tmp/xgb_matplotlib_cache'))
    from . import train_xgb as legacy
    import xgboost,sklearn
    prices,paths=data_audit(legacy,out);breadth=breadth_frames(prices)
    ranks=legacy.build_cross_sectional_ranks([legacy.DATA_DIR/f'{t}.csv' for t in TICKERS])
    previous_protocol=json.loads((PREVIOUS/'protocolo.json').read_text(encoding='utf-8'))
    # Cache de referencia aceito somente com as mesmas entradas/codigo/protocolo.
    for name,digest in {**previous_protocol['source_hashes'],**previous_protocol['protected_baseline_hashes']}.items():
        if hashlib.sha256((ROOT/name).read_bytes()).hexdigest()!=digest:raise ValueError(f'Baseline/entrada mudou: {name}')
    versions=dict(xgboost=xgboost.__version__,sklearn=sklearn.__version__,numpy=np.__version__,pandas=pd.__version__)
    for key in versions:
        if versions[key]!=previous_protocol['versions'][key]:raise ValueError('Versao mudou: '+key)
    source_paths=paths+[Path(__file__),Path(__file__).with_name('features_experimento.py'),PREVIOUS/'protocolo.json']
    previous_choices=pd.read_csv(PREVIOUS/'escolhas_validacao_2024.csv').set_index(['ticker','horizon']).chosen
    protocol=dict(groups=GROUPS,parameters=ANCHOR,max_rounds=400,threads_per_model=2,
        versions=versions,tickers=args.tickers,horizons=args.horizons,folds=[list(f) for f in FOLDS],
        selection='Menor RMSE 2024 por ativo/horizonte; todos os grupos com os mesmos parametros.',
        pruning='Sem y; variaveis constantes, mais de 10% ausentes, ou Spearman absoluto acima de 0.98; somente fit purgado.',
        heldout_status='2025 e diagnostico retrospectivo. 2026 ate marco ja entrou em analises OOF anteriores; nenhum bloco inteiramente intocado no acervo.',
        confirmation='Congelar candidata antes de obter/avaliar dados de periodo ainda nao inspecionado; sem alegacao de teste independente nesta rodada.',
        source_hashes={str(p.relative_to(ROOT)):hashlib.sha256(p.read_bytes()).hexdigest() for p in source_paths})
    manifest=out/'protocolo.json'
    if args.resume and manifest.exists() and json.loads(manifest.read_text(encoding='utf-8'))!=protocol:raise ValueError('Protocolo mudou')
    manifest.write_text(json.dumps(protocol,ensure_ascii=False,indent=2),encoding='utf-8')
    if args.audit_only:
        records=[]
        for ticker in args.tickers:
            frame,pools=augmented(ticker,5,ranks,breadth,legacy)
            records+=audit_original_features(frame,pools['completo'],ticker)
        pd.DataFrame(records).to_csv(out/'redundancia_original.csv',index=False)
        print(pd.read_csv(out/'auditoria_precos.csv').to_string(index=False))
        print('Pares altamente correlacionados:',len(records),'duplicados numericos:',sum(r['duplicate_numeric'] for r in records))
        return
    started=time.monotonic()
    def run_case(ticker,h):
        csv=out/'cases'/f'{ticker}_h{h}.csv';ap=out/'cases'/f'{ticker}_h{h}_audit.json'
        if args.resume and csv.exists() and ap.exists():return json.loads(ap.read_text())
        frame,pools=augmented(ticker,h,ranks,breadth,legacy)
        cached=pd.read_csv(PREVIOUS/'cases'/f'{ticker}_h{h}.csv',parse_dates=['date']).set_index('date')
        y=frame.actual.to_numpy(float);frames=[];audits=[]
        for fold,start,end,phase in FOLDS:
            tr,te=masks(frame,start,end,phase)
            g=frame.iloc[te][['date','maturity','actual','current']].copy()
            reference=cached.loc[g.date]
            np.testing.assert_allclose(reference.actual,g.actual,rtol=0,atol=1e-14)
            g['completo']=reference[ANCHOR['name']].to_numpy()
            g['previous']=reference[previous_choices.loc[(ticker,h)]].to_numpy()
            g['zero']=0.;g['historical']=reference.historical.to_numpy()
            cut=int(len(tr)*.8);iv=tr[cut:];fit=tr[:cut]
            fit=fit[(frame.maturity.iloc[fit]<frame.date.iloc[iv[0]]).to_numpy()]
            # Checagem numerica da referencia reutilizada, com todo o mesmo protocolo.
            if ticker==TICKERS[0] and h==5 and fold in ['val_2024H1','test_2025Q4']:
                predicted,_=fit_predict(frame[pools['completo']].to_numpy(np.float32),y,frame,tr,te,ANCHOR,400,2)
                np.testing.assert_allclose(predicted,g.completo,rtol=0,atol=1e-12)
            for group in GROUPS[1:]:
                columns,removed=select_columns(frame,fit,pools[group])
                X=frame[columns].to_numpy(np.float32)
                p,audit=fit_predict(X,y,frame,tr,te,ANCHOR,400,2)
                g[group]=p
                audits.append(dict(ticker=ticker,horizon=h,fold=fold,phase=phase,group=group,
                    columns=columns,removed=removed,n_features=len(columns),first_prediction=str(g.date.min().date()),
                    last_prediction=str(g.date.max().date()),n_predict=len(g),**audit))
            frames.append(g.assign(ticker=ticker,horizon=h,fold=fold,phase=phase))
        pd.concat(frames).to_csv(csv,index=False)
        ap.write_text(json.dumps(audits,indent=2),encoding='utf-8')
        print(f'{ticker} h={h}: tres conjuntos novos + referencia; {time.monotonic()-started:.1f}s',flush=True)
        return audits
    all_audits=[]
    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        jobs=[pool.submit(run_case,t,h) for t in args.tickers for h in args.horizons]
        for job in as_completed(jobs):all_audits+=job.result()
    flat=[{**{k:v for k,v in r.items() if k not in ['columns','removed']},
           'columns':';'.join(r['columns']),'removed_count':len(r['removed'])} for r in all_audits]
    pd.DataFrame(flat).to_csv(out/'auditoria_temporal_features.csv',index=False)
    for name,digest in {**previous_protocol['source_hashes'],**previous_protocol['protected_baseline_hashes']}.items():
        if hashlib.sha256((ROOT/name).read_bytes()).hexdigest()!=digest:raise AssertionError('Referencia original alterada')
    summarize(out)
    (out/'verificacao_baseline.json').write_text(json.dumps(dict(unchanged=True,anchor_checks=2,audits=len(all_audits)),indent=2),encoding='utf-8')


if __name__=='__main__':main()
