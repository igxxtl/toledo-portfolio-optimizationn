"""Inventario completo OOF e comparacao pareada das camadas do experimento atual."""
from pathlib import Path
import json
import hashlib
import os
import shutil
import numpy as np
import pandas as pd
from .camadas_intermediarias import ROOT, OUT
from .avaliar_predicao_xgb import fmt, table
from intel.Blacklitterman import simular_dois_cenarios as engine
from intel.Blacklitterman import config as cfg

DEST=OUT/'todas_previsoes'
TCC=Path(r'C:\Users\igor.toledo_visagio\Documents\TCC 2')


def measure(y,p):
    y,p=np.asarray(y,float),np.asarray(p,float)
    e=p-y; mse=float(np.mean(e*e)); variance=float(np.var(y))
    std_y,std_p=float(np.std(y)),float(np.std(p))
    return dict(n=len(y),mae=float(np.mean(abs(e))),rmse=float(np.sqrt(mse)),mse=mse,
        hr=float(np.mean(np.sign(y)==np.sign(p))*100),r2=1-mse/variance if variance>0 else np.nan,
        pearson=float(np.corrcoef(y,p)[0,1]) if std_y>0 and std_p>0 else np.nan,
        nrmse=float(np.sqrt(mse)/std_y) if std_y>0 else np.nan,
        bias=float(np.mean(e)),std_real=std_y,std_pred=std_p,
        amplitude_ratio=std_p/std_y if std_y>0 else np.nan,
        skill_zero=1-mse/float(np.mean(y*y)) if np.mean(y*y)>0 else np.nan)


def figure(name,caption):
    return '\n'.join([r'\begin{figure}[H]\centering', '\\caption{'+caption+'}',
        r'\includegraphics[width=0.95\textwidth]{05_Resultados/predicao_xgb/'+name+'}',
        r'\par\smallskip{\footnotesize Fonte: elaboração própria.}\end{figure}'])


def main():
    DEST.mkdir(parents=True,exist_ok=True)
    cache=ROOT/'tmp/xgb_matplotlib_cache'; cache.mkdir(parents=True,exist_ok=True)
    os.environ.setdefault('MPLCONFIGDIR',str(cache))
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    meta=pd.read_csv(ROOT/'criacao_modelo_xgb/metricas_todos_horizontes.csv')
    records,yearly,folds,obs,inputs=[],[],[],[],[]
    aliases=[]; reconciliations=[]
    for row in meta.itertuples():
        path=ROOT/f'criacao_modelo_xgb/{row.ticker}_h{row.horizon}d_predicoes.csv'
        inputs.append(path)
        d=pd.read_csv(path); d.date=pd.to_datetime(d.date)
        if d.date.duplicated().any() or d[['y_real','y_pred']].isna().any().any(): raise ValueError(path)
        tags=dict(asset=row.ticker,horizon=row.horizon,source=row.q_source)
        score=measure(d.y_real,d.y_pred)
        if not np.isclose(score['hr'],row.dir_acc_pct):
            raise ValueError(f'Acuracia salva nao reproduzida: {path}')
        reconciliations.append(dict(**tags,rmse_saved=row.rmse_wf,rmse_recomputed=score['rmse'],
            mae_saved=row.mae_wf,mae_recomputed=score['mae'],r2_saved=row.r2_wf,r2_recomputed=score['r2'],
            error_metrics_match=bool(np.isclose(score['rmse'],row.rmse_wf,rtol=1e-6))))
        records.append(dict(**tags,first=str(d.date.min().date()),last=str(d.date.max().date()),**score))
        for year,g in d.groupby(d.date.dt.year): yearly.append(dict(**tags,year=int(year),**measure(g.y_real,g.y_pred)))
        for fold,g in d.groupby('fold'): folds.append(dict(**tags,fold=int(fold),**measure(g.y_real,g.y_pred)))
        d=d.assign(**tags,absolute_error=(d.y_pred-d.y_real).abs(),squared_error=(d.y_pred-d.y_real)**2,
                   direction_hit=(np.sign(d.y_real)==np.sign(d.y_pred)))
        obs.append(d)
        if row.vencedor:
            alias=ROOT/f'criacao_modelo_xgb/{row.ticker}_predicoes.csv'
            alias_df=pd.read_csv(alias)
            aliases.append(dict(asset=row.ticker,horizon=row.horizon,
                identical=bool(np.allclose(alias_df.y_pred,d.y_pred) and len(alias_df)==len(d))))
    all_obs=pd.concat(obs,ignore_index=True); complete=pd.DataFrame(records)
    pd.DataFrame(reconciliations).to_csv(DEST/'reconciliacao_metricas_antigas.csv',index=False)
    for name,data in [('xgb_todas_observacoes',all_obs),('xgb_60_modelos',complete),('xgb_por_ano',pd.DataFrame(yearly)),
                      ('xgb_por_fold',pd.DataFrame(folds)),('aliases_vencedores',pd.DataFrame(aliases))]:
        data.to_csv(DEST/f'{name}.csv',index=False)
    metrics=['hr','r2','nrmse','pearson','amplitude_ratio']
    complete[metrics].corr(method='spearman').to_csv(DEST/'relacoes_metricas_spearman.csv')
    complete[metrics].corr().to_csv(DEST/'relacoes_metricas_pearson.csv')
    agg=complete.groupby('horizon').agg(models=('asset','size'),predictions=('n','sum'),
        hr=('hr','mean'),r2=('r2','mean'),rmse=('rmse','mean'),mae=('mae','mean'),
        nrmse=('nrmse','mean'),negative_r2=('r2',lambda s:int((s<0).sum())))
    agg.to_csv(DEST/'xgb_resumo_horizontes.csv')
    # Mesmo motor/rodada: evita misturar posterior historico diario com Q de outra configuracao.
    prices,_,_=engine.load_inputs()
    audit=pd.read_csv(OUT/'auditoria.csv'); audit=audit[audit.variant.eq('hybrid')].copy()
    paired=[]; excluded=[]
    rf=(1+cfg.SELIC_ANNUAL)**(1/252)-1
    for r in audit.itertuples():
        date=pd.Timestamp(r.signal_date); position=prices.index.get_loc(date); h=int(r.horizon)
        if date!=pd.Timestamp(r.prediction_date):
            excluded.append(dict(asset=r.asset,date=r.signal_date,reason='previsao desatualizada'));continue
        end=position+h
        if end>=len(prices) or prices.index[end]>pd.Timestamp('2025-12-30'):
            excluded.append(dict(asset=r.asset,date=r.signal_date,reason='alvo termina apos 2025-12-30'));continue
        actual=np.log(prices.iloc[end][r.asset]/prices.iloc[position][r.asset])/h-rf
        next_day=prices.iloc[position+1][r.asset]/prices.iloc[position][r.asset]-1-rf
        paired.append(dict(asset=r.asset,date=r.signal_date,horizon=h,maturity=str(prices.index[end].date()),
            actual_daily_equivalent_excess=actual,actual_next_day_excess=next_day,
            XGB=r.prediction/h-rf,Q=r.Q,Prior=r.prior,Posterior=r.posterior,
            omega=r.omega,confidence=r.confidence,news_count=r.news_count,gate_active=r.gate_active))
    chain=pd.DataFrame(paired)
    chain.to_csv(DEST/'camadas_pares.csv',index=False)
    pd.DataFrame(excluded).to_csv(DEST/'camadas_exclusoes.csv',index=False)
    chain_scores=[]
    for target in ['actual_daily_equivalent_excess','actual_next_day_excess']:
        for scope,g in [('todos',chain),*[(a,g) for a,g in chain.groupby('asset')]]:
            for layer in ['XGB','Q','Prior','Posterior']:
                chain_scores.append(dict(target=target,scope=scope,layer=layer,**measure(g[target],g[layer])))
    cs=pd.DataFrame(chain_scores);cs.to_csv(DEST/'camadas_metricas.csv',index=False)
    chain[['XGB','Q','Prior','Posterior']].corr().to_csv(DEST/'camadas_correlacoes.csv')
    relation=[]
    for a,b in [('XGB','Q'),('Q','Posterior'),('Prior','Posterior')]:
        relation.append(dict(first=a,second=b,n=len(chain),mean_absolute_change=float((chain[a]-chain[b]).abs().mean()),
            direction_agreement=float((np.sign(chain[a])==np.sign(chain[b])).mean()*100),
            pearson=float(chain[a].corr(chain[b]))))
    pd.DataFrame(relation).to_csv(DEST/'camadas_transformacoes.csv',index=False)
    plt.rcParams.update({'font.family':'DejaVu Sans','font.size':10,'axes.spines.top':False,'axes.spines.right':False})
    fig,ax=plt.subplots(figsize=(8,5))
    for h,g in complete.groupby('horizon'): ax.scatter(g.hr,g.r2,label=f'{h} pregões',s=36,alpha=.8)
    ax.axhline(0,color='#777',lw=.8);ax.axvline(50,color='#aaa',lw=.8,ls='--')
    ax.set_xlabel('Acurácia direcional OOF (%)');ax.set_ylabel('R² OOF');ax.legend()
    ax.set_title('Relação entre acerto da direção e erro de magnitude\n60 combinações de ativo e horizonte · histórico completo')
    fig.tight_layout();fig.savefig(DEST/'relacao_acuracia_r2.png',dpi=180);plt.close(fig)
    fig,axes=plt.subplots(1,2,figsize=(10,4.4))
    for ax,a,b in [(axes[0],'XGB','Q'),(axes[1],'Prior','Posterior')]:
        x,y=chain[a]*10000,chain[b]*10000
        ax.scatter(x,y,s=15,alpha=.55);lo=min(x.min(),y.min());hi=max(x.max(),y.max())
        ax.plot([lo,hi],[lo,hi],color='#999',ls='--',lw=1)
        ax.set_xlabel(a+' (pontos-base/dia)');ax.set_ylabel(b+' (pontos-base/dia)')
    fig.suptitle('Transformação das previsões nas decisões mensais de 2025')
    fig.tight_layout();fig.savefig(DEST/'relacao_camadas.png',dpi=180);plt.close(fig)
    inventory=dict(xgb_files=len(complete),xgb_forecasts=len(all_obs),first=str(all_obs.date.min().date()),
        last=str(all_obs.date.max().date()),years=sorted(all_obs.date.dt.year.unique().tolist()),
        aliases_excluded=len(aliases),monthly_candidates=len(audit),monthly_scored=len(chain),monthly_excluded=len(excluded),
        metadata_sources=meta.q_source.value_counts().to_dict(),
        old_error_metrics_mismatches=sum(not r['error_metrics_match'] for r in reconciliations),
        limitation='Series individuais dos regressores/classificadores perdedores nao foram preservadas; apenas acuracias agregadas.',
        legacy_daily_Q_status='CSV legado Q/Omega contem bytes invalidos; nao mesclar com a rodada mensal atual.',
        source_hashes={str(p.relative_to(ROOT)):hashlib.sha256(p.read_bytes()).hexdigest() for p in inputs+[OUT/'auditoria.csv',Path(__file__)]})
    (DEST/'manifest.json').write_text(json.dumps(inventory,ensure_ascii=False,indent=2),encoding='utf-8')
    # Complemento salvo para rastreabilidade; editora recebe o fragmento existente, no mesmo caminho.
    chosen=cs[cs.scope.eq('todos') & cs.target.eq('actual_daily_equivalent_excess')]
    nxt=cs[cs.scope.eq('todos') & cs.target.eq('actual_next_day_excess')]
    rho=complete.hr.corr(complete.r2)
    lines=[r'\subsection{Inventário de todas as previsões quantitativas disponíveis}',
        f'O inventário contém {len(complete)} combinações de vinte ativos e três horizontes (5, 10 e 21 pregões), '
        f'totalizando {len(all_obs):,} previsões OOF entre {inventory["first"]} e {inventory["last"]}. '
        'Os vinte arquivos de vencedores são aliases das séries por horizonte e não são contados novamente. '
        'Cada previsão possui data, retorno realizado, retorno estimado, erro e fold. '
        'As métricas abaixo utilizam os alvos originais salvos no treinamento, enquanto o teste financeiro de 2025 é um recorte específico discutido anteriormente.',
        table('Métricas do histórico OOF completo por horizonte (médias entre ativos).',
            ['h','Previsões',r'HR (\%)',r'MAE$\times100$',r'RMSE$\times100$',r'R$^2$',r'NRMSE'],
            [[h,int(r.predictions),fmt(r.hr),fmt(r.mae*100),fmt(r.rmse*100),fmt(r.r2,3),fmt(r.nrmse,3)] for h,r in agg.iterrows()], 'rrrrrrr'),
        'MAE e RMSE estão multiplicados por 100 em unidades de log-retorno, e não correspondem exatamente a percentuais de retorno simples. '
        'São médias simples das métricas dos vinte ativos; não se calcula um R-quadrado agregado misturando as séries. '
        'Todos os resultados individuais seguem na tabela completa e nos arquivos por ano e por fold. '
        'A extensão até 2026 descreve o acervo de treinamento e não amplia retroativamente o período do experimento financeiro de 2025.',
        f'A reconciliação identificou {sum(not r["error_metrics_match"] for r in reconciliations)} combinações em que as métricas de erro do resumo histórico '
        'não correspondem às previsões finais salvas. O código calcula MAE, RMSE e R-quadrado do regressor antes de escolher o classificador, '
        'mas pode exportar a previsão do classificador vencedor sem atualizar essas métricas. '
        'Por isso, as tabelas abaixo recalculam todas as métricas a partir dos pares y-real/y-pred efetivamente armazenados, '
        'sem alterar os arquivos originais. A acurácia direcional do resumo foi reproduzida nas sessenta combinações.',
        r'\subsection{Como as métricas se relacionam}',
        r'Para o erro $e_t=\widehat y_t-y_t$, MAE é a média de $|e_t|$, MSE é a média de $e_t^2$ e RMSE é a raiz do MSE. '
        'O RMSE dá maior peso a erros grandes e é pelo menos tão alto quanto o MAE na mesma amostra. '
        'A acurácia direcional ignora o tamanho do erro: uma previsão pode acertar alta ou baixa e ainda errar fortemente sua intensidade.',
        r'Com a variância populacional dos alvos $\sigma_y^2$, tem-se $R^2=1-\mathrm{MSE}/\sigma_y^2$ e '
        r'$\mathrm{NRMSE}=\mathrm{RMSE}/\sigma_y=\sqrt{1-R^2}$. '
        'Assim, R-quadrado e NRMSE expressam a mesma informação de erro normalizado, em escalas distintas. '
        'R-quadrado negativo significa erro superior ao da média retrospectiva dos alvos da amostra; essa média não é uma previsão negociável.',
        r'A correlação de Pearson mede associação linear entre previsão e resultado, mas não sua calibração. '
        r'Se $r=\mathrm{corr}(y,\widehat y)$, $k=\sigma_{\widehat y}/\sigma_y$ e $b=\mathrm{média}(\widehat y-y)/\sigma_y$, '
        r'então $R^2=2rk-k^2-b^2$. '
        'Mesmo correlação positiva pode coexistir com R-quadrado negativo se a amplitude ou o viés estiverem inadequados.',
        figure('relacao_acuracia_r2.png','Relação entre qualidade da direção e da magnitude no histórico OOF.'),
        f'Entre as sessenta combinações, a correlação de Pearson entre acurácia direcional e R-quadrado é {fmt(rho,3)}. '
        'Essa associação é descritiva: as combinações compartilham datas e ativos e não constituem sessenta experimentos independentes. '
        'Não há identidade matemática entre acurácia e R-quadrado, diferentemente da relação entre NRMSE e R-quadrado.',
        r'\subsection{Métricas individuais de todas as combinações}',
        'As tabelas por horizonte apresentam toda a cobertura, incluindo a correlação entre previsão e alvo (r). '
        'Os registros individuais, métricas por ano e por fold foram preservados para rastrear mudanças de desempenho. '
        'Os arquivos armazenam a previsão do estimador escolhido em cada combinação. '
        'O resumo histórico contém acurácia de regressor e classificador, mas não preserva ambas as séries perdedoras; '
        'não é possível reconstruir seus MAE, RMSE ou R-quadrado separadamente sem nova execução.']
    for h,g in complete.groupby('horizon'):
        lines.append(table(f'Todas as previsões OOF: horizonte de {h} pregões.',
            ['Ativo','N',r'HR (\%)',r'MAE$\times100$',r'RMSE$\times100$',r'R$^2$','r'],
            [[r.asset,r.n,fmt(r.hr),fmt(r.mae*100),fmt(r.rmse*100),fmt(r.r2,3),fmt(r.pearson,3)] for r in g.sort_values('asset').itertuples()], 'lrrrrrr'))
    lines += [r'\subsection{Como as previsões se transformam entre as camadas}',
        r'A sequência quantitativa é $\widehat r_{XGB}\rightarrow Q\rightarrow\mu_{BL}$, combinada com o prior $\pi$ e a incerteza $\Omega$. '
        'O XGBoost estima o log-retorno no horizonte selecionado. Q mistura esse sinal com o componente textual. '
        'O posterior combina Q com o prior por meio da covariância e da confiança. '
        'Prior, Q e posterior não são previsores treinados para exatamente a mesma tarefa; sua avaliação requer uma convenção comum explícita.',
        f'Usa-se a auditoria da rodada mensal atual: {len(audit)} pares ativo-data candidatos e {len(chain)} com alvo concluído até 30/12/2025. '
        f'Foram excluídos {len(excluded)} pares sem desfecho elegível. Todas as camadas são comparadas exatamente nos mesmos pares. '
        'Não se misturam os exports do antigo pipeline diário com os resultados atuais, pois possuem configurações distintas e o CSV legado de Q/Omega apresenta bytes inválidos.',
        r'Na primeira comparação, o alvo é o log-retorno realizado no horizonte h dividido por h, menos a referência diária livre de risco. '
        r'A previsão XGB recebe a mesma transformação $\widehat r_h/h-r_f$. '
        'Q, prior e posterior são utilizados na escala diária de excesso implementada pelo motor. '
        'A compatibilização entre log-retorno e o prior baseado em retornos simples é aproximada. '
        'O resultado mede alinhamento com retorno diário equivalente, não uma previsão independente de cada pregão nem uma estratégia financeira.',
        table('Comparação pareada das camadas: alvo diário equivalente no horizonte de cada ativo.',
              ['Camada','N',r'HR excesso (\%)',r'MAE (pb)',r'RMSE (pb)',r'R$^2$','r'],
              [[r.layer,r.n,fmt(r.hr),fmt(r.mae*10000),fmt(r.rmse*10000),fmt(r.r2,3),fmt(r.pearson,3)] for r in chosen.itertuples()], 'lrrrrrr'),
        'Os erros estão em pontos-base por dia (valor vezes 10.000). A acurácia de excesso compara sinais após descontar a referência diária e não coincide com a acurácia de retorno bruto apresentada nas tabelas do XGBoost.',
        table('Sensibilidade da avaliação ao alvo: excesso de retorno simples no pregão seguinte ao sinal.',
              ['Camada','N',r'HR excesso (\%)',r'MAE (pb)',r'RMSE (pb)',r'R$^2$'],
              [[r.layer,r.n,fmt(r.hr),fmt(r.mae*10000),fmt(r.rmse*10000),fmt(r.r2,3)] for r in nxt.itertuples()], 'lrrrrr'),
        'A segunda comparação mantém os mesmos pares e altera somente o alvo para o retorno simples do fechamento do sinal ao fechamento seguinte, '
        'menos a referência diária. Ela testa alinhamento com a frequência diária do BL, mas não respeita o alvo nativo multiperíodo do XGBoost. '
        'Não se compara diretamente uma métrica dessa tabela com a da anterior como se o problema preditivo fosse o mesmo.',
        figure('relacao_camadas.png','Relação entre as previsões das camadas nas mesmas decisões mensais.'),
        table('Transformação dos sinais nas mesmas observações.',
              ['Transição',r'Variação média absoluta (pb)',r'Mesmo sinal (\%)','r'],
              [[r['first']+' / '+r['second'],fmt(r['mean_absolute_change']*10000),fmt(r['direction_agreement']),fmt(r['pearson'],3)] for r in relation], 'lrrr'),
        'A diagonal nos gráficos corresponde a previsões idênticas entre camadas. Pontos próximos dela para prior e posterior indicam pouca atualização, '
        'não maior qualidade preditiva por si só. A incerteza e a covariância determinam a força da atualização. '
        'Uma correlação elevada entre camadas tampouco demonstra correlação elevada com retornos realizados.',
        r'\subsection{Alcance da avaliação conjunta}',
        'A comparação permite separar três perguntas: qual sinal foi previsto, qual magnitude foi estimada e quanto a integração alterou a previsão. '
        'O histórico completo, os recortes anuais e os folds são descrições dos artefatos existentes. '
        'A escolha original entre regressor e classificador e os ajustes de hiperparâmetros não foram refeitos de forma aninhada. '
        'Horizontes sobrepostos, empresas relacionadas e configurações selecionadas retrospectivamente limitam a independência das observações. '
        'Nenhuma dessas métricas, isoladamente, comprova rentabilidade ou ganho causal de uma camada.']
    fragment='\n\n'.join(lines)+'\n'
    controlled=OUT/'qualidade_camadas/secao.tex'
    if controlled.exists():
        begin=fragment.index(r'\subsection{Como as previsões se transformam entre as camadas}')
        end=fragment.index(r'\subsection{Alcance da avaliação conjunta}')
        fragment=fragment[:begin]+controlled.read_text(encoding='utf-8')+'\n'+fragment[end:]
    (DEST/'predicao_xgb.tex').write_text(fragment,encoding='utf-8')
    # Edita o documento ja aberto, sem criar outro documento no editor.
    (TCC/'05_Resultados/predicao_xgb.tex').write_text(fragment,encoding='utf-8')
    for p in DEST.glob('*.png'): shutil.copyfile(p,TCC/'05_Resultados/predicao_xgb'/p.name)
    print(json.dumps(inventory,ensure_ascii=False,indent=2)[:2200])
    print(chosen[['layer','n','hr','rmse','r2']].to_string(index=False))


if __name__=='__main__':main()
