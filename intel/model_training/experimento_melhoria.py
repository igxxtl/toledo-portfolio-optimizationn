"""Experimento isolado de regressao; nunca altera criacao_modelo_xgb.

python -m intel.model_training.experimento_melhoria
Selecao: 2024. Comparacao retrospectiva congelada: 2025. Mesmos alvos em cada par.
"""
from __future__ import annotations
import argparse
import hashlib
import json
import os
import time
import warnings
from concurrent.futures import ThreadPoolExecutor,as_completed
from pathlib import Path
import numpy as np
import pandas as pd

ROOT=Path(__file__).resolve().parents[2]
DEFAULT_OUT=ROOT/'analise_resultados/melhoria_xgb'
TICKERS=['ABEV3','B3SA3','BBAS3','BBDC4','CMIG4','EQTL3','GGBR4','ITUB4','LREN3','MGLU3','PETR3','PETR4','PRIO3','RADL3','RAIL3','SANB11','SBSP3','SUZB3','VALE3','WEGE3']
CONFIGS=[dict(name=f'sq_{profile}_w{weight}',objective='reg:squarederror',profile=profile,weight=weight)
         for profile in ['regular','conservador'] for weight in [0,1,5]]
CONFIGS += [dict(name=f'abs_conservador_w{weight}',objective='reg:absoluteerror',profile='conservador',weight=weight) for weight in [0,1]]
FOLDS=[('val_2024H1','2024-01-01','2024-07-01','validation'),
       ('val_2024H2','2024-07-01','2025-01-01','validation'),
       ('test_2025Q1','2025-01-01','2025-04-01','test'),
       ('test_2025Q2','2025-04-01','2025-07-01','test'),
       ('test_2025Q3','2025-07-01','2025-10-01','test'),
       ('test_2025Q4','2025-10-01','2026-01-01','test')]


def metric(y,p):
    y,p=np.asarray(y,float),np.asarray(p,float)
    error=p-y; mse=float(np.mean(error**2)); v=float(np.var(y));zero=float(np.mean(y**2))
    return dict(n=len(y),rmse=float(np.sqrt(mse)),mae=float(np.mean(abs(error))),
                r2=1-mse/v if v else np.nan,hr=float(np.mean(np.sign(y)==np.sign(p))*100),
                skill_zero=1-mse/zero if zero else np.nan)


def masks(frame,start,end,phase):
    start,end=pd.Timestamp(start),pd.Timestamp(end)
    train=(frame.date<start)&(frame.maturity<start)
    test=(frame.date>=start)&(frame.date<end)&frame.current.notna()
    # Validacao exige todos os desfechos conhecidos antes de 2025.
    cutoff=pd.Timestamp('2025-01-01') if phase=='validation' else pd.Timestamp('2025-12-31')
    test &= frame.maturity<cutoff
    return np.flatnonzero(train),np.flatnonzero(test)


def fit_predict(X,y,frame,tr,te,config,max_rounds,threads):
    from xgboost import XGBRegressor
    if len(tr)<200 or len(te)==0:raise ValueError('Amostra insuficiente')
    cut=int(len(tr)*.8);iv=tr[cut:]
    # Purga independente na separacao fit/early stopping: nenhum alvo cruza a fronteira.
    fit=tr[:cut];fit=fit[(frame.maturity.iloc[fit]<frame.date.iloc[iv[0]]).to_numpy()]
    if len(fit)<100:raise ValueError('Fit insuficiente apos purga')
    params=dict(objective=config['objective'],eval_metric='rmse',tree_method='hist',
                learning_rate=.03,subsample=.8,colsample_bytree=.8,max_bin=64,
                max_depth=4,min_child_weight=10,reg_alpha=.1,reg_lambda=1.,
                n_jobs=threads,random_state=42,verbosity=0)
    if config['profile']=='conservador':
        params.update(max_depth=2,min_child_weight=30,reg_alpha=.5,reg_lambda=10.)
    def weights(n):
        w=np.exp(config['weight']*np.arange(n)/n);return w/w.mean()
    center=float(y[fit].mean());scale=max(float(y[fit].std()),1e-8)
    early=XGBRegressor(**params,n_estimators=max_rounds,early_stopping_rounds=20)
    early.fit(X[fit],(y[fit]-center)/scale,sample_weight=weights(len(fit)),
              eval_set=[(X[iv],(y[iv]-center)/scale)],verbose=False)
    rounds=int(early.best_iteration)+1
    # Refit somente com todos os alvos conhecidos no inicio do fold.
    center=float(y[tr].mean());scale=max(float(y[tr].std()),1e-8)
    model=XGBRegressor(**params,n_estimators=rounds)
    model.fit(X[tr],(y[tr]-center)/scale,sample_weight=weights(len(tr)),verbose=False)
    pred=model.predict(X[te]).astype(float)*scale+center
    audit=dict(n_train=len(tr),n_fit=len(fit),n_early_stop=len(iv),rounds=rounds,
        max_train_maturity=str(frame.maturity.iloc[tr].max().date()),
        early_stop_start=str(frame.date.iloc[iv[0]].date()),
        max_fit_maturity=str(frame.maturity.iloc[fit].max().date()))
    return pred,audit


def prepare(ticker,h,ranks,legacy):
    raw=legacy.carregar_df(legacy.DATA_DIR/f'{ticker}.csv')
    if ticker in ranks:
        for col in ranks[ticker]:raw[col]=ranks[ticker][col].reindex(pd.to_datetime(raw.date)).to_numpy()
    with warnings.catch_warnings():
        warnings.simplefilter('ignore',pd.errors.PerformanceWarning)
        features,columns=legacy.build_features(raw)
    features=features.copy()
    features['date']=pd.to_datetime(raw.date).to_numpy()
    features['maturity']=pd.to_datetime(raw.date).shift(-h).to_numpy()
    features['actual']=np.log(raw.close.shift(-h)/raw.close).to_numpy()
    current=pd.read_csv(legacy.OUT_DIR/f'{ticker}_h{h}d_predicoes.csv',parse_dates=['date']).set_index('date')
    features['current']=features.date.map(current.y_pred)
    keep=features[columns+['actual','maturity']].notna().all(axis=1)
    features=features[keep].reset_index(drop=True)
    if not np.isfinite(features[columns].to_numpy()).all():raise ValueError('Features nao finitas')
    return features,columns


def summarize(output):
    cases=sorted((output/'cases').glob('*.csv'))
    predictions=pd.concat([pd.read_csv(p) for p in cases],ignore_index=True)
    predictions.to_csv(output/'previsoes.csv',index=False)
    rows=[]
    for (ticker,h,phase),g in predictions.groupby(['ticker','horizon','phase']):
        for col in ['current','zero','historical']+[c['name'] for c in CONFIGS]:
            rows.append(dict(ticker=ticker,horizon=h,phase=phase,model=col,**metric(g.actual,g[col])))
    all_metrics=pd.DataFrame(rows)
    all_metrics.to_csv(output/'metricas_todas.csv',index=False)
    choices=[];comparisons=[]
    for (ticker,h),validation in all_metrics[all_metrics.phase.eq('validation')].groupby(['ticker','horizon']):
        candidates=validation[validation.model.isin([c['name'] for c in CONFIGS])].sort_values(['rmse','model'])
        best=candidates.iloc[0]
        choices.append(dict(ticker=ticker,horizon=h,chosen=best.model,validation_rmse=best.rmse,
                            validation_current_rmse=float(validation[validation.model.eq('current')].rmse.iloc[0])))
        tests=all_metrics[(all_metrics.ticker==ticker)&(all_metrics.horizon==h)&all_metrics.phase.eq('test')].set_index('model')
        current=tests.loc['current'];chosen=tests.loc[best.model]
        comparisons.append(dict(ticker=ticker,horizon=h,chosen=best.model,n=int(chosen.n),
            rmse_current=current.rmse,rmse_new=chosen.rmse,rmse_reduction_pct=(1-chosen.rmse/current.rmse)*100,
            mae_current=current.mae,mae_new=chosen.mae,mae_reduction_pct=(1-chosen.mae/current.mae)*100,
            hr_current=current.hr,hr_new=chosen.hr,r2_current=current.r2,r2_new=chosen.r2,
            skill_zero_current=current.skill_zero,skill_zero_new=chosen.skill_zero,
            rmse_historical=tests.loc['historical'].rmse,rmse_zero=tests.loc['zero'].rmse))
    choice=pd.DataFrame(choices);comp=pd.DataFrame(comparisons)
    choice.to_csv(output/'escolhas_validacao_2024.csv',index=False)
    comp.to_csv(output/'comparacao_pareada_2025.csv',index=False)
    # Configuracoes fixas todas apresentadas: nao escolher pelo ranking do teste.
    fixed=all_metrics[all_metrics.phase.eq('test')].groupby(['horizon','model'],as_index=False).agg(
        assets=('ticker','size'),n=('n','sum'),rmse_mean=('rmse','mean'),mae_mean=('mae','mean'),
        hr_mean=('hr','mean'),r2_mean=('r2','mean'),skill_zero_mean=('skill_zero','mean'))
    fixed.to_csv(output/'configuracoes_fixas_teste.csv',index=False)
    summary=comp.groupby('horizon',as_index=False).agg(assets=('ticker','size'),n=('n','sum'),
        rmse_current=('rmse_current','mean'),rmse_new=('rmse_new','mean'),
        mae_current=('mae_current','mean'),mae_new=('mae_new','mean'),
        hr_current=('hr_current','mean'),hr_new=('hr_new','mean'),
        skill_zero_current=('skill_zero_current','mean'),skill_zero_new=('skill_zero_new','mean'),
        wins_rmse=('rmse_reduction_pct',lambda s:int((s>0).sum())),
        beat_zero=('skill_zero_new',lambda s:int((s>0).sum())))
    summary['rmse_reduction_pct']=(1-summary.rmse_new/summary.rmse_current)*100
    summary.to_csv(output/'resumo_horizontes.csv',index=False)
    # Mantem o horizonte atual por ticker para comparar a entrada do sistema.
    existing=ROOT/'intel/Blacklitterman/outputs/dois_cenarios/horizontes_validacao.csv'
    old_choices=pd.read_csv(existing)
    old_choices['ticker']=old_choices.asset.str.replace('.SA','',regex=False)
    kept=comp.merge(old_choices[['ticker','horizon']],on=['ticker','horizon'],validate='one_to_one')
    kept.to_csv(output/'comparacao_horizontes_atuais.csv',index=False)
    # Confianca simples: não gerar um vencedor por 2025; apenas reportar candidato escolhido antes dele.
    report=dict(cases=len(comp),forecast_pairs_test=int(all_metrics[(all_metrics.phase=='test')&(all_metrics.model=='current')].n.sum()),
        chosen_counts=choice.chosen.value_counts().to_dict(),
        wins_vs_current=int((comp.rmse_new<comp.rmse_current).sum()),
        wins_vs_zero=int((comp.rmse_new<comp.rmse_zero).sum()),
        wins_vs_historical=int((comp.rmse_new<comp.rmse_historical).sum()),
        current_horizon=dict(assets=len(kept),rmse_current=float(kept.rmse_current.mean()),
            rmse_new=float(kept.rmse_new.mean()),mae_current=float(kept.mae_current.mean()),mae_new=float(kept.mae_new.mean()),
            hr_current=float(kept.hr_current.mean()),hr_new=float(kept.hr_new.mean()),
            wins_rmse=int((kept.rmse_new<kept.rmse_current).sum())),
        caution='2025 ja foi examinado na pesquisa: resultado retrospectivo; nao e novo teste independente.')
    (output/'resultado.json').write_text(json.dumps(report,ensure_ascii=False,indent=2),encoding='utf-8')
    print(summary.to_string(index=False),flush=True)
    print(json.dumps(report,ensure_ascii=False,indent=2),flush=True)


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--tickers',nargs='+',default=TICKERS)
    parser.add_argument('--horizons',nargs='+',type=int,default=[5,10,21])
    parser.add_argument('--output',type=Path,default=DEFAULT_OUT)
    parser.add_argument('--max-rounds',type=int,default=400)
    parser.add_argument('--threads',type=int,default=2)
    parser.add_argument('--workers',type=int,default=4)
    parser.add_argument('--resume',action='store_true')
    args=parser.parse_args()
    args.output.mkdir(parents=True,exist_ok=True);(args.output/'cases').mkdir(exist_ok=True)
    os.environ.setdefault('MPLCONFIGDIR',str(ROOT/'tmp/xgb_matplotlib_cache'))
    from . import train_xgb as legacy
    import xgboost,sklearn,optuna
    assets=[legacy.DATA_DIR/f'{t}.csv' for t in TICKERS]
    ranks=legacy.build_cross_sectional_ranks(assets)
    # Hash antes e depois para garantir que o historico nao foi sobrescrito.
    protected=sorted(legacy.OUT_DIR.glob('*_predicoes.csv'))+sorted(legacy.OUT_DIR.glob('metricas*.csv'))
    protected_hashes={str(p.relative_to(ROOT)):hashlib.sha256(p.read_bytes()).hexdigest() for p in protected}
    sources=assets+[legacy.DATA_DIR/'IBOV.csv',legacy.DATA_DIR/'USDBRL.csv',Path(__file__),Path(legacy.__file__)]
    protocol=dict(configurations=CONFIGS,folds=[list(f) for f in FOLDS],max_rounds=args.max_rounds,threads=args.threads,
        horizons=args.horizons,tickers=args.tickers,features='Mesmas features do treinador, ranks dos mesmos 20 ativos.',
        selection='Menor RMSE em 2024 por ativo e horizonte; sem mudar horizonte usando 2025.',
        refit='Trimestral em 2025, somente alvos maturados antes do inicio do fold.',
        purge='Por data de maturidade, tanto treino/teste quanto fit/early stopping.',
        metrics='Alvos recalculados dos precos locais; atual e candidatas nos mesmos pares.',
        versions=dict(xgboost=xgboost.__version__,sklearn=sklearn.__version__,numpy=np.__version__,pandas=pd.__version__,optuna=optuna.__version__),
        source_hashes={str(p.relative_to(ROOT)):hashlib.sha256(p.read_bytes()).hexdigest() for p in sources},
        protected_baseline_hashes=protected_hashes)
    manifest=args.output/'protocolo.json'
    if args.resume and manifest.exists() and json.loads(manifest.read_text(encoding='utf-8'))!=protocol:
        raise ValueError('Protocolo/entradas mudaram; nao reutilizar casos antigos')
    manifest.write_text(json.dumps(protocol,ensure_ascii=False,indent=2),encoding='utf-8')
    audit_rows=[];started=time.monotonic()
    def run_case(ticker,h):
            output=args.output/'cases'/f'{ticker}_h{h}.csv'
            audit_path=args.output/'cases'/f'{ticker}_h{h}_audit.json'
            if args.resume and output.exists() and audit_path.exists():
                return json.loads(audit_path.read_text(encoding='utf-8'))
            frame,features=prepare(ticker,h,ranks,legacy)
            X=frame[features].to_numpy(dtype=np.float32);y=frame.actual.to_numpy(float)
            # Bases observaveis por data, nao pela media de 2025.
            realized=pd.Series(np.log(pd.read_csv(legacy.DATA_DIR/f'{ticker}.csv').close.shift(-h)/pd.read_csv(legacy.DATA_DIR/f'{ticker}.csv').close))
            raw_dates=pd.to_datetime(pd.read_csv(legacy.DATA_DIR/f'{ticker}.csv').date)
            history=pd.Series(realized.shift(h).expanding(min_periods=80).mean().to_numpy(),index=raw_dates)
            fold_frames=[];case_audit=[]
            for name,start,end,phase in FOLDS:
                tr,te=masks(frame,start,end,phase)
                g=frame.iloc[te][['date','maturity','actual','current']].copy()
                g['historical']=g.date.map(history);g['zero']=0.
                g=g.assign(ticker=ticker,horizon=h,phase=phase,fold=name)
                for config in CONFIGS:
                    p,audit=fit_predict(X,y,frame,tr,te,config,args.max_rounds,args.threads)
                    g[config['name']]=p
                    case_audit.append(dict(ticker=ticker,horizon=h,fold=name,phase=phase,model=config['name'],
                        first_prediction=str(frame.date.iloc[te].min().date()),last_prediction=str(frame.date.iloc[te].max().date()),
                        n_predict=len(te),features=len(features),**audit))
                fold_frames.append(g)
            pd.concat(fold_frames).to_csv(output,index=False)
            audit_path.write_text(json.dumps(case_audit,indent=2),encoding='utf-8')
            print(f'{ticker} h={h}: 8 configuracoes x 6 folds concluidos; {time.monotonic()-started:.1f}s',flush=True)
            return case_audit
    # Treinos independentes: arquivos de saida distintos, features e ranks somente leitura.
    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        futures=[pool.submit(run_case,t,h) for t in args.tickers for h in args.horizons]
        for future in as_completed(futures):audit_rows+=future.result()
    pd.DataFrame(audit_rows).to_csv(args.output/'auditoria_temporal.csv',index=False)
    for p in protected:
        if hashlib.sha256(p.read_bytes()).hexdigest()!=protected_hashes[str(p.relative_to(ROOT))]:raise AssertionError('Baseline alterado')
    # Audit JSONs nao sao incluidos no glob CSV.
    summarize(args.output)
    (args.output/'verificacao_baseline.json').write_text(json.dumps(dict(unchanged=True,files=len(protected)),indent=2),encoding='utf-8')


if __name__=='__main__':main()
