"""Qualidade preditiva e efeito controlado do sentimento, sem retreino/tuning."""
from pathlib import Path
import hashlib
import json
import os
import numpy as np
import pandas as pd
from .camadas_intermediarias import ROOT, OUT
from .todas_previsoes import measure
from .avaliar_predicao_xgb import fmt, table
from intel.Blacklitterman import simular_dois_cenarios as engine
from intel.Blacklitterman import config as cfg

DEST=OUT/'qualidade_camadas'
TEX=Path(r'C:\Users\igor.toledo_visagio\Documents\TCC 2\05_Resultados\predicao_xgb.tex')
LAYERS=['XGB','Q_neutro','Q','BL_XGB','BL_neutro','BL','Prior']
NAMES=dict(XGB='XGBoost',Q_neutro='Q sem polaridade',Q='Q completo',BL_XGB='BL com XGB',
           BL_neutro='BL sem polaridade',BL='BL completo',Prior='Prior')


def score(y,p):
    result=measure(y,p)
    sign=np.sign(y);pred=np.sign(p)
    result['balanced_accuracy']=float(np.mean([np.mean(pred[sign==k]==k) for k in np.unique(sign)])*100)
    return result


def main():
    DEST.mkdir(parents=True,exist_ok=True)
    prices,returns,paths=engine.load_inputs()
    predictions,choices,pred_paths=engine.prepare_predictions(prices,pd.Timestamp('2025-01-01'))
    dates,signals,audit=engine.build_signals(returns,prices,predictions,pd.Timestamp('2025-01-01'),pd.Timestamp('2025-12-31'),126,3.,'hybrid')
    # Exige reproducao da auditoria anterior antes da intervencao.
    old=pd.read_csv(OUT/'auditoria.csv');old=old[old.variant.eq('hybrid')]
    for col in ['Q','prior','posterior','omega']:
        np.testing.assert_allclose(audit[col],old[col],rtol=1e-10,atol=1e-12)
    rf=(1+cfg.SELIC_ANNUAL)**(1/252)-1
    frames=[]
    for signal,g in audit.groupby('signal_date',sort=True):
        g=g.set_index('asset').loc[prices.columns].copy()
        g.index.name='asset'
        cov=returns.loc[:signal].tail(126).cov().to_numpy()+np.eye(len(g))*1e-7
        prior_cov=cfg.TAU*cov
        coverage=np.minimum(1.,g.news_count.to_numpy()/cfg.NEWS_COUNT_MIN)
        text_delta=(1-cfg.ALPHA_Q)*coverage*g.sentiment_component.to_numpy()/g.horizon.to_numpy()
        neutral_q=g.Q.to_numpy()-text_delta
        xgb_q=g.prediction.to_numpy()/g.horizon.to_numpy()-rf
        pi=g.prior.to_numpy();omega=g.omega.to_numpy()
        def posterior(q):
            return pi+prior_cov@np.linalg.solve(prior_cov+np.diag(omega),q-pi)
        np.testing.assert_allclose(posterior(g.Q.to_numpy()),g.posterior.to_numpy(),rtol=1e-10,atol=1e-12)
        g['XGB']=xgb_q;g['Q_neutro']=neutral_q;g['BL_XGB']=posterior(xgb_q)
        g['BL_neutro']=posterior(neutral_q);g['BL']=g.posterior;g['Prior']=pi
        g['text_delta']=text_delta;g['signal_date']=signal
        frames.append(g.reset_index())
    data=pd.concat(frames,ignore_index=True)
    observed=[];excluded=[]
    for r in data.itertuples():
        signal=pd.Timestamp(r.signal_date);pos=prices.index.get_loc(signal);h=int(r.horizon)
        if signal!=pd.Timestamp(r.prediction_date):raise ValueError('Previsao nao coincide com sinal')
        result=dict(r._asdict());result.pop('Index',None)
        result['actual_next_day']=prices.iloc[pos+1][r.asset]/prices.iloc[pos][r.asset]-1-rf
        if prices.index[pos+h]<=pd.Timestamp('2025-12-30'):
            result['actual_horizon_daily']=np.log(prices.iloc[pos+h][r.asset]/prices.iloc[pos][r.asset])/h-rf
            result['maturity']=prices.index[pos+h]
        else:
            result['actual_horizon_daily']=np.nan
            excluded.append(dict(asset=r.asset,signal_date=signal,reason='desfecho do horizonte apos 30/12/2025'))
        # Conversao para a metrica de retorno simples no pregão seguinte.
        for layer in LAYERS:
            result[layer+'_next_day']=(np.expm1(result[layer]+rf)-rf) if layer in ['XGB','Q_neutro','Q'] else result[layer]
        observed.append(result)
    data=pd.DataFrame(observed)
    data.to_csv(DEST/'previsoes_e_resultados.csv',index=False)
    pd.DataFrame(excluded).to_csv(DEST/'exclusoes_horizonte.csv',index=False)
    rows=[]
    for target in ['actual_horizon_daily','actual_next_day']:
        valid=data.dropna(subset=[target])
        groups=[('todos',valid),('com_noticias',valid[valid.news_count>0]),
                ('sem_noticias',valid[valid.news_count==0]),('gate_ativo',valid[valid.gate_active]),
                *[(a,g) for a,g in valid.groupby('asset')]]
        for scope,g in groups:
            if g.empty:continue
            for layer in LAYERS:
                p=g[layer+'_next_day'] if target=='actual_next_day' else g[layer]
                rows.append(dict(target=target,scope=scope,layer=layer,**score(g[target].to_numpy(),p.to_numpy())))
    scores=pd.DataFrame(rows);scores.to_csv(DEST/'scores.csv',index=False)
    # Variacoes de erro pareadas: positivo significa melhora ao adicionar o componente.
    changes=[]
    for target in ['actual_horizon_daily','actual_next_day']:
        valid=data.dropna(subset=[target])
        for scope,g in [('todos',valid),('com_noticias',valid[valid.news_count>0]),('gate_ativo',valid[valid.gate_active])]:
            if g.empty:continue
            for before,after in [('XGB','Q'),('Q_neutro','Q'),('BL_neutro','BL'),('XGB','BL_XGB'),('Q','BL'),('Prior','BL')]:
                b=g[before+'_next_day'] if target=='actual_next_day' else g[before]
                a=g[after+'_next_day'] if target=='actual_next_day' else g[after]
                y=g[target]; eb=(b-y)**2;ea=(a-y)**2
                changes.append(dict(target=target,scope=scope,before=before,after=after,n=len(g),
                    mse_reduction_pct=float((1-ea.mean()/eb.mean())*100),
                    mae_reduction_pb=float(((b-y).abs().mean()-(a-y).abs().mean())*10000),
                    hr_gain_pp=float(((np.sign(a)==np.sign(y)).mean()-(np.sign(b)==np.sign(y)).mean())*100),
                    squared_error_wins=int((ea<eb-1e-20).sum()),squared_error_ties=int(np.isclose(ea,eb,rtol=0,atol=1e-20).sum())))
    changes=pd.DataFrame(changes);changes.to_csv(DEST/'efeito_pareado.csv',index=False)
    main_scores=scores[(scores.target=='actual_horizon_daily')&(scores.scope=='todos')].copy()
    main_scores['rank_rmse']=main_scores.rmse.rank(method='min').astype(int)
    main_scores['rank_hr']=main_scores.hr.rank(ascending=False,method='min').astype(int)
    main_scores.to_csv(DEST/'ranking_horizonte.csv',index=False)
    direct=scores[(scores.target=='actual_next_day')&(scores.scope=='todos')]
    n=int(main_scores.n.iloc[0]); news=int(data[data.actual_horizon_daily.notna()].news_count.gt(0).sum())
    gates=int(data[data.actual_horizon_daily.notna()].gate_active.sum())
    lines=[r'\subsection{Avaliação preditiva de cada camada e contribuição das notícias}',
        'Esta análise confronta cada previsão com o retorno realizado. Seu objetivo é responder se o componente textual '
        'melhora Q e se a atualização Black--Litterman melhora a previsão. Não utiliza retorno financeiro de carteiras como substituto de qualidade preditiva.',
        'São comparadas sete configurações nas mesmas observações: XGBoost; Q completo; Q sem polaridade textual; '
        'BL alimentado somente pelo XGBoost; BL sem polaridade; BL completo; e prior sem views. '
        'No controle sem polaridade, o sentimento é fixado em zero, mas a contagem de notícias, a cobertura, o peso do sinal quantitativo, '
        'a matriz de incerteza, o gate e o prior permanecem iguais. Assim, a diferença contra a versão completa isola '
        'a contribuição numérica da polaridade textual, condicionada ao mecanismo de cobertura existente. '
        'Ela não representa ausência total de notícias, pois a quantidade de manchetes continua participando da confiança.',
        'O controle BL com XGB utiliza o retorno quantitativo como Q, mas mantém a matriz de incerteza e o gate da rodada original. '
        'Sua comparação com o XGBoost isola a atualização pelo prior e pela covariância sob essa confiança. '
        'Não se remove o gate nesse contraste, pois isso alteraria simultaneamente outra parte do sistema.',
        f'A avaliação no horizonte nativo utiliza {n} pares ativo-data de 2025, dos quais {news} possuem notícias e {gates} atingem o gate. '
        'O alvo é o log-retorno realizado em h pregões dividido por h, menos a referência diária livre de risco. '
        'O XGBoost e Q recebem a mesma escala; prior e posterior mantêm a escala diária do motor. '
        'A aproximação entre log-retornos e retornos aritméticos do BL permanece explícita. '
        'As classificações são do sinal do excesso de retorno, e não do retorno bruto.',
        r'O score de magnitude é $S=1-\mathrm{MSE}_{modelo}/\mathrm{MSE}_{zero}$, usando zero excesso como referência '
        '(equivalente à referência livre de risco nesta escala). Valor positivo indica melhora de erro contra essa referência, '
        'zero indica empate e valor negativo indica piora. Não é uma nota arbitrária de zero a cem. '
        'A acurácia balanceada é a média da proporção de acertos em cada classe de sinal presente, reduzindo a influência do desequilíbrio entre altas e baixas.',
        render_scores(main_scores,'Scores preditivos no mesmo alvo e nas mesmas observações.'),
        'MAE e RMSE estão em pontos-base por dia. Menor erro é melhor; maior acurácia, acurácia balanceada, R-quadrado e score é melhor. '
        'Esses critérios podem apontar vencedores distintos, pois acertar o sinal não garante estimar corretamente a magnitude.',
        render_effects(changes,'todos','actual_horizon_daily','Efeito incremental pareado no horizonte selecionado.'),
        'Na tabela de efeitos, redução positiva de MSE indica melhora e ganho positivo de HR indica mais acertos de sinal. '
        'XGB para Q mede a mistura completa e não isola sentimento: a cobertura também muda o peso quantitativo. '
        'Q sem polaridade para Q completo e BL sem polaridade para BL completo são os controles específicos da polaridade textual.',
        render_effects(changes,'com_noticias','actual_horizon_daily','Efeito restrito às observações que possuem manchetes.'),
        f'Os efeitos com gate ativo também foram exportados, mas envolvem apenas {gates} pares e são estritamente descritivos. '
        'Mesmo com um gate ativo em um papel, a covariância pode transmitir a mudança a outros ativos no mesmo rebalanceamento. '
        'Por isso, o efeito textual sobre o posterior não precisa estar restrito aos ativos com manchetes.',
        r'\subsection{Sensibilidade à definição de previsão diária}',
        'Como o BL trabalha na frequência diária, calcula-se também o erro contra o retorno simples do pregão seguinte ao sinal, '
        'com todos os 240 pares mensais. Para XGB e Q, a previsão de log-retorno diário equivalente é convertida '
        'em retorno simples por exponenciação; o posterior e o prior são utilizados como retornos simples excedentes. '
        'Essa conversão supõe uma taxa diária equivalente constante e não transforma o XGBoost em um modelo treinado para um pregão. '
        'O horizonte nativo e o pregão seguinte são tarefas distintas; seus scores não devem ser misturados.',
        render_scores(direct,'Scores contra o retorno simples excedente no pregão seguinte.'),
        'As comparações são retrospectivas e limitadas às decisões mensais do experimento. '
        'Os horizontes e a configuração do BL não foram escolhidos novamente por esses scores. '
        'A amostra possui dependência temporal e entre ativos; diferenças pequenas não são declaradas estatisticamente significativas. '
        'O efeito controlado indica o que mudou nesta execução ao retirar a polaridade, não uma garantia de contribuição preditiva em outro período.']
    # Conclusao em linguagem de previsao, derivada das metricas calculadas.
    best_error=main_scores.loc[main_scores.rmse.idxmin()];best_hr=main_scores.loc[main_scores.hr.idxmax()]
    effect_text=changes[(changes.scope=='todos')&(changes.target=='actual_horizon_daily')]
    q_effect=effect_text[(effect_text.before=='Q_neutro')&(effect_text.after=='Q')].iloc[0]
    bl_effect=effect_text[(effect_text.before=='BL_neutro')&(effect_text.after=='BL')].iloc[0]
    lines.append(f'A polaridade textual altera o MSE de Q em uma redução de {fmt(q_effect.mse_reduction_pct)}\\% '
        f'e o do posterior em {fmt(bl_effect.mse_reduction_pct)}\\%. '
        'O sinal negativo no primeiro resultado representa piora de magnitude em Q. '
        'Neste experimento, o efeito textual não é uniformemente favorável: sua combinação com a atualização BL apresenta resultado diferente da mistura direta em Q.')
    lines.append(f'Neste alvo, o menor RMSE pertence a {NAMES[best_error.layer]}, com {fmt(best_error.rmse*10000)} pontos-base, '
        f'e a maior acurácia pertence a {NAMES[best_hr.layer]}, com {fmt(best_hr.hr)}\\%. '
        'As comparações específicas de sentimento e atualização BL devem ser lidas separadamente para identificar a origem dessa diferença.')
    cache=ROOT/'tmp/xgb_matplotlib_cache';cache.mkdir(parents=True,exist_ok=True)
    os.environ.setdefault('MPLCONFIGDIR',str(cache))
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    chart=main_scores.set_index('layer').loc[LAYERS]
    fig,axes=plt.subplots(1,2,figsize=(11,5))
    names=[NAMES[k] for k in LAYERS]
    axes[0].barh(names,chart.rmse*10000,color='#286888');axes[0].invert_yaxis()
    axes[0].set_xlabel('RMSE (pontos-base/dia) · menor é melhor')
    axes[1].barh(names,chart.balanced_accuracy,color='#438b70');axes[1].invert_yaxis()
    axes[1].axvline(50,color='#777',ls='--',lw=1);axes[1].set_xlim(0,100)
    axes[1].set_xlabel('Acurácia balanceada (%) · maior é melhor')
    fig.suptitle(f'Qualidade das previsões · mesmos {n} pares e alvo no horizonte selecionado')
    fig.tight_layout();fig.savefig(DEST/'scores_camadas.png',dpi=180);plt.close(fig)
    import shutil
    shutil.copyfile(DEST/'scores_camadas.png',TEX.parent/'predicao_xgb/scores_camadas.png')
    lines.insert(10, '\n'.join([r'\begin{figure}[H]\centering',
        r'\caption{Comparação de erro e acerto balanceado das previsões de cada camada.}',
        r'\includegraphics[width=\textwidth]{05_Resultados/predicao_xgb/scores_camadas.png}',
        r'\par\smallskip{\footnotesize Fonte: elaboração própria.}\end{figure}']))
    fragment='\n\n'.join(lines)+'\n'
    (DEST/'secao.tex').write_text(fragment,encoding='utf-8')
    current=TEX.read_text(encoding='utf-8')
    start=r'\subsection{Como as previsões se transformam entre as camadas}'
    new=r'\subsection{Avaliação preditiva de cada camada e contribuição das notícias}'
    end=r'\subsection{Alcance da avaliação conjunta}'
    a=current.find(start) if start in current else current.find(new)
    b=current.find(end)
    if a<0 or b<a: raise ValueError('Limites da secao nao encontrados')
    TEX.write_text(current[:a]+fragment+'\n'+current[b:],encoding='utf-8')
    paths+=pred_paths+[OUT/'auditoria.csv',Path(__file__)]
    (DEST/'manifest.json').write_text(json.dumps(dict(n_monthly=len(data),n_native=n,news_native=news,gate_native=gates,
        controls='Zero somente polaridade; mesma cobertura, omega, gate, prior e covariancia.',
        risk_free_annual=cfg.SELIC_ANNUAL,news_gate=cfg.NEWS_COUNT_MIN,
        sha256={str(p.relative_to(ROOT)):hashlib.sha256(p.read_bytes()).hexdigest() for p in paths}),indent=2),encoding='utf-8')
    print(main_scores[['layer','n','hr','balanced_accuracy','rmse','skill_zero','r2']].to_string(index=False))
    print(changes[(changes.scope=='todos')&(changes.target=='actual_horizon_daily')].to_string(index=False))


def render_scores(g,caption):
    return table(caption,['Previsão','N',r'HR (\%)',r'BA (\%)','MAE','RMSE',r'$100S$',r'R$^2$'],
        [[NAMES[r.layer],r.n,fmt(r.hr),fmt(r.balanced_accuracy),fmt(r.mae*10000),fmt(r.rmse*10000),fmt(r.skill_zero*100),fmt(r.r2,3)] for r in g.itertuples()], 'lrrrrrrr')


def render_effects(changes,scope,target,caption):
    selected=changes[(changes.scope==scope)&(changes.target==target)]
    return table(caption,['Comparação','N',r'Redução MSE (\%)','Ganho HR (p.p.)'],
        [[NAMES[r.before]+' / '+NAMES[r.after],r.n,fmt(r.mse_reduction_pct),fmt(r.hr_gain_pp)] for r in selected.itertuples()], 'lrrr')


if __name__=='__main__':main()
