"""Avalia artefatos XGB existentes; nao retreina nem escolhe modelos usando 2025.

python -m analise_resultados.avaliar_predicao_xgb
"""
from pathlib import Path
import hashlib
import json
import os
import shutil
import numpy as np
import pandas as pd
from .camadas_intermediarias import OUT, ROOT
from intel.Blacklitterman import simular_dois_cenarios as engine

DEST = OUT / 'predicao_xgb'
TCC = Path(r'C:\Users\igor.toledo_visagio\Documents\TCC 2')
LABELS = np.array([-1., 0., 1.])


def baselines(actual, h):
    """Em t, somente alvos iniciados ate t-h estao maturados ao fechamento."""
    known = actual.shift(h)
    mean = known.expanding(min_periods=80).mean()
    counts = pd.concat([(np.sign(known) == k).cumsum() for k in LABELS], axis=1)
    majority = pd.Series(LABELS[counts.to_numpy().argmax(axis=1)], index=actual.index)
    majority[known.notna().cumsum() < 80] = np.nan
    return mean, majority


def metrics(g):
    y, p = g.actual.to_numpy(), g.pred.to_numpy()
    mse = np.mean((p-y)**2)
    rmse_zero = float(np.sqrt(np.mean(y**2)))
    rmse_hist = float(np.sqrt(np.mean((g.historical-y)**2)))
    recalls = [float(np.mean(np.sign(p)[np.sign(y)==c] == c)) for c in LABELS if (np.sign(y)==c).any()]
    return dict(n=len(g), rmse=float(np.sqrt(mse)), mae=float(np.mean(np.abs(p-y))),
        r2=float(1-np.sum((p-y)**2)/np.sum((y-y.mean())**2)),
        hr=float(np.mean(np.sign(p)==np.sign(y))*100),
        balanced_accuracy=float(np.mean(recalls)*100),
        hr_historical=float(np.mean(np.sign(g.historical)==np.sign(y))*100),
        hr_majority=float(np.mean(g.majority==np.sign(y))*100),
        rmse_zero=rmse_zero, rmse_historical=rmse_hist,
        skill_zero=float(1-mse/rmse_zero**2), skill_historical=float(1-mse/rmse_hist**2),
        bias=float(np.mean(p-y)), pred_std=float(np.std(p)), actual_std=float(np.std(y)))


def table(caption, headings, records, spec):
    return '\n'.join([r'\begin{table}[H]\centering', '\\caption{'+caption+'}',
        r'{\footnotesize\begin{tabular}{'+spec+'}\\hline', ' & '.join(headings)+r'\\ \hline',
        *[' & '.join(map(str,r))+r'\\' for r in records],
        r'\hline\end{tabular}}\par\smallskip{\footnotesize Fonte: elaboração própria.}\end{table}'])


def fmt(v, n=2):
    return f'{v:.{n}f}'.replace('.', ',')


def main():
    cache = ROOT / 'tmp/xgb_matplotlib_cache'
    cache.mkdir(parents=True, exist_ok=True)
    os.environ.setdefault('MPLCONFIGDIR', str(cache))
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    DEST.mkdir(parents=True, exist_ok=True)
    prices, _, inputs = engine.load_inputs()
    selected, choices, paths = engine.prepare_predictions(prices, pd.Timestamp('2025-01-01'))
    inputs += paths + [Path(__file__)]
    observations, scores, shap = [], [], []
    for asset in prices:
        base = asset.removesuffix('.SA')
        for h in (5, 10, 21):
            path = ROOT / f'criacao_modelo_xgb/{base}_h{h}d_predicoes.csv'
            prediction = pd.read_csv(path, usecols=['date','y_pred'])
            prediction.date = pd.to_datetime(prediction.date)
            if prediction.date.duplicated().any():
                raise ValueError(f'Datas duplicadas: {path}')
            pred = prediction.set_index('date').y_pred.reindex(prices.index)
            actual = np.log(prices[asset].shift(-h)/prices[asset])
            maturity = pd.Series(prices.index,index=prices.index).shift(-h)
            historical, majority = baselines(actual,h)
            frame = pd.DataFrame(dict(actual=actual,pred=pred,historical=historical,majority=majority,maturity=maturity))
            mask = ((frame.index.year==2025) & maturity.le(pd.Timestamp('2025-12-30')))
            frame = frame[mask].dropna()
            tags = dict(asset=base,horizon=h,selected=h==selected[asset]['h'])
            observations.append(frame.reset_index(names='date').assign(**tags))
            for period,g in [('2025',frame), *[(str(q),g) for q,g in frame.groupby(frame.index.to_period('Q'))]]:
                scores.append(dict(**tags,period=period,**metrics(g)))
        h = selected[asset]['h']
        path = ROOT / f'criacao_modelo_xgb/{base}_h{h}d_shap.csv'
        if path.exists():
            f = pd.read_csv(path)
            f['relative_importance'] = f.shap_mean_abs/f.shap_mean_abs.sum()
            shap.append(f.assign(asset=base,horizon=h))
            inputs.append(path)
    obs, score = pd.concat(observations),pd.DataFrame(scores)
    obs.to_csv(DEST/'observacoes.csv',index=False)
    score.to_csv(DEST/'metricas.csv',index=False)
    chosen = score[score.selected & score.period.eq('2025')].sort_values('asset')
    quarterly = score[score.selected & ~score.period.eq('2025')].groupby('period').agg(
        assets=('asset','size'),hr=('hr','mean'),hr_majority=('hr_majority','mean'),
        hr_historical=('hr_historical','mean'),skill_zero=('skill_zero','mean'))
    quarterly.to_csv(DEST/'trimestres_selecionados.csv')
    horizons = score[score.period.eq('2025')].groupby('horizon').agg(
        hr=('hr','mean'),hr_majority=('hr_majority','mean'),skill_zero=('skill_zero','mean'),
        beat_zero=('skill_zero',lambda s:int((s>0).sum())))
    horizons.to_csv(DEST/'horizontes_diagnostico.csv')
    # Mesma unidade em cada comparacao; skill normaliza erro pelo benchmark do ativo/horizonte.
    plt.rcParams.update({'font.family':'DejaVu Sans','font.size':10,'axes.spines.top':False,'axes.spines.right':False})
    pivot = score[score.period.eq('2025')].pivot(index='asset',columns='horizon',values='skill_zero')*100
    fig,ax=plt.subplots(figsize=(7.6,8.5))
    limit=max(abs(pivot.to_numpy()).max(),1)
    im=ax.imshow(pivot,aspect='auto',cmap='RdBu',vmin=-limit,vmax=limit)
    ax.set_xticks(range(3),['5 pregões','10 pregões','21 pregões']);ax.set_yticks(range(len(pivot)),pivot.index)
    for i in range(len(pivot)):
        for j in range(3):
            value=pivot.iloc[i,j]
            ax.text(j,i,f'{value:.1f}',ha='center',va='center',color='white' if abs(value)>limit*.55 else '#182c3b',fontsize=8)
    ax.set_title('Redução do erro quadrático frente ao retorno zero (%)\nPositivo: XGB melhora a referência; negativo: piora',pad=14)
    fig.colorbar(im,ax=ax,label='Skill × 100');fig.tight_layout();fig.savefig(DEST/'skill_horizontes.png',dpi=180);plt.close(fig)
    fig,ax=plt.subplots(figsize=(8.5,4.5))
    for col,label in [('hr','XGBoost'),('hr_majority','Direção majoritária histórica'),('hr_historical','Sinal da média histórica')]:
        ax.plot(quarterly.index,quarterly[col],marker='o',label=label)
    ax.set_ylabel('Acurácia direcional média entre ativos (%)');ax.set_ylim(0,100)
    ax.set_title('Estabilidade trimestral · horizontes escolhidos em 2024');ax.legend(fontsize=9)
    ax.grid(axis='y',alpha=.2);fig.tight_layout();fig.savefig(DEST/'estabilidade_trimestral.png',dpi=180);plt.close(fig)
    features=pd.concat(shap)
    features.to_csv(DEST/'shap_historico.csv',index=False)
    # Completa features ausentes com zero para a media ter sempre 20 ativos no denominador.
    importance=features.pivot(index='asset',columns='feature',values='relative_importance').fillna(0).mean().sort_values(ascending=False)
    importance.rename('mean_relative_importance').to_csv(DEST/'shap_relativo_medio.csv')
    fig,ax=plt.subplots(figsize=(8.5,5))
    top=importance.head(10).sort_values()*100
    ax.barh(top.index,top.values,color='#286888');ax.set_xlabel('Participação média no SHAP absoluto por ativo (%)')
    ax.set_title('Atributos dos artefatos históricos · horizontes selecionados\nDiagnóstico de modelos salvos, sem recorte exclusivo de 2025',fontsize=11)
    fig.tight_layout();fig.savefig(DEST/'shap_historico.png',dpi=180);plt.close(fig)
    summary=dict(n_assets=len(chosen),n_observations_selected=int(chosen.n.sum()),
        hr_mean=float(chosen.hr.mean()),hr_majority_mean=float(chosen.hr_majority.mean()),
        hr_historical_mean=float(chosen.hr_historical.mean()),
        beat_majority=int((chosen.hr>chosen.hr_majority+1e-10).sum()),
        tie_majority=int(np.isclose(chosen.hr,chosen.hr_majority,atol=1e-10,rtol=0).sum()),
        beat_zero_rmse=int((chosen.skill_zero>0).sum()),beat_historical_rmse=int((chosen.skill_historical>0).sum()),
        negative_r2=int((chosen.r2<0).sum()), best_hr=chosen.loc[chosen.hr.idxmax(),'asset'],
        worst_hr=chosen.loc[chosen.hr.idxmin(),'asset'], top_features=importance.head(5).to_dict())
    manifest=dict(summary=summary,baseline='expanding, >=80 alvos maturados; fechamento t admite alvo maturado em t',
        majority_tie_rule='-1, 0, +1 nesta ordem; argmax deterministico',
        test_end='2025-12-30',selection='Horizonte por acuracia 2024; nao alterado apos diagnostico de 2025',
        limitation='Artefatos preexistentes: escolha regressor/classificador e tuning nao foram refeitos de forma aninhada.',
        sha256={str(p.relative_to(ROOT)):hashlib.sha256(p.read_bytes()).hexdigest() for p in inputs})
    (DEST/'manifest.json').write_text(json.dumps(manifest,ensure_ascii=False,indent=2),encoding='utf-8')
    text = [r'\subsection{Comparação das previsões com referências observáveis}',
        'A avaliação complementar utiliza os mesmos artefatos quantitativos e os horizontes escolhidos em 2024. '
        'O alvo é o log-retorno acumulado em 5, 10 ou 21 pregões. Para cada ativo e horizonte, '
        'o modelo e as referências são avaliados nos mesmos pares, com desfechos terminando até 30 de dezembro de 2025. '
        'Não houve retreinamento ou escolha de novos modelos usando os resultados abaixo.',
        'São utilizadas três referências: retorno zero; média expansiva dos log-retornos de mesmo horizonte já realizados; '
        'e direção mais frequente entre esses alvos históricos. As duas referências históricas exigem pelo menos 80 alvos conhecidos. '
        'Na previsão ao fechamento de t, o último alvo admitido é aquele iniciado em t-h e concluído em t. '
        'A direção majoritária é somente uma regra de classificação; não lhe é atribuída magnitude ou RMSE. '
        'Diferentemente da maior classe observada em 2025, apresentada anteriormente como diagnóstico ex post, '
        'esta referência é calculável sem consultar desfechos futuros.',
        r'Define-se $S_0=1-\mathrm{MSE}_{XGB}/\mathrm{MSE}_{0}$. Valores positivos representam melhora contra o retorno zero. '
        r'O indicador $S_{hist}$ substitui o denominador pelo MSE da média histórica. '
        'Essa comparação respeita a escala de cada ativo e horizonte; RMSEs brutos de horizontes distintos não são comparados como problemas equivalentes.',
        table('Qualidade preditiva dos horizontes selecionados em 2024.',
              ['Ativo','h','HR XGB','HR maj.','HR média',r'$100S_0$',r'$100S_{hist}$'],
              [[r.asset,r.horizon,fmt(r.hr),fmt(r.hr_majority),fmt(r.hr_historical),fmt(100*r.skill_zero),fmt(100*r.skill_historical)] for r in chosen.itertuples()], 'lrrrrrr'),
        f'A acurácia média entre ativos é {fmt(summary["hr_mean"])}\\%, frente a {fmt(summary["hr_majority_mean"])}\\% '
        f'para a direção majoritária histórica e {fmt(summary["hr_historical_mean"])}\\% para o sinal da média histórica. '
        f'O XGBoost supera estritamente a direção majoritária em {summary["beat_majority"]} ativos e empata em {summary["tie_majority"]}. '
        f'Em magnitude, melhora o RMSE do retorno zero em {summary["beat_zero_rmse"]} ativos e o da média histórica em {summary["beat_historical_rmse"]}. '
        'Os resultados distinguem capacidade de acertar o sinal e capacidade de estimar a magnitude. Valores de HR e skills na tabela estão em porcentagem.',
        'LREN3, PETR4, RADL3, RAIL3 e WEGE3 superam a direção majoritária histórica. '
        'WEGE3 também reduz o erro frente às duas referências de magnitude. '
        'SBSP3 apresenta elevada acurácia, mas empata com a direção majoritária; isso mostra por que um percentual alto de acertos, isoladamente, '
        'não certifica ganho preditivo. O skill frente ao retorno zero e o R-quadrado usam referências diferentes: '
        'é possível superar o retorno zero e ainda apresentar R-quadrado negativo frente à média retrospectiva do teste.',
        r'\subsection{Horizontes e estabilidade temporal}',
        table('Diagnóstico de todos os horizontes em 2025, sem nova seleção.',
            ['h',r'HR XGB (\%)',r'HR maj. (\%)',r'$100S_0$ médio','Ativos com melhora'],
            [[h,fmt(r.hr),fmt(r.hr_majority),fmt(100*r.skill_zero),int(r.beat_zero)] for h,r in horizons.iterrows()], 'rrrrr'),
        'Os três horizontes são mostrados como diagnóstico retrospectivo. Seu desempenho em 2025 não é utilizado para substituir '
        'a escolha feita em 2024. Cada horizonte possui quantidade diferente de alvos elegíveis e graus distintos de sobreposição.',
        figure('skill_horizontes.png','Heterogeneidade do erro preditivo por ativo e horizonte.'),
        table('Estabilidade trimestral dos horizontes previamente selecionados.',
              ['Trimestre',r'HR XGB (\%)',r'HR maj. (\%)',r'HR média (\%)'],
              [[q,fmt(r.hr),fmt(r.hr_majority),fmt(r.hr_historical)] for q,r in quarterly.iterrows()], 'lrrr'),
        figure('estabilidade_trimestral.png','Comparação trimestral com referências disponíveis no instante da previsão.'),
        f'No segundo trimestre, a acurácia média do XGBoost é {fmt(quarterly.loc["2025Q2", "hr"])}\\%, '
        f'contra {fmt(quarterly.loc["2025Q2", "hr_majority"])}\\% da direção majoritária histórica. '
        f'No quarto trimestre, o modelo alcança {fmt(quarterly.loc["2025Q4", "hr"])}\\%, '
        f'enquanto essa referência atinge {fmt(quarterly.loc["2025Q4", "hr_majority"])}\\%. '
        'Essa mudança evidencia dependência temporal do desempenho, sem permitir identificar sua causa somente pelas métricas agregadas.',
        'As médias trimestrais atribuem o mesmo peso a cada ativo. O trimestre corresponde à data de origem da previsão; '
        'um alvo pode terminar no trimestre seguinte, desde que concluído até o final do teste. '
        'Os alvos sobrepostos e os choques comuns entre ações impedem tratar os pares como observações independentes. '
        'As diferenças apresentadas são descritivas, sem interpretação automática de significância estatística.',
        r'\subsection{Atributos e limites da explicabilidade}',
        figure('shap_historico.png','Participação relativa dos atributos nos resumos SHAP históricos.'),
        'Nos resumos disponíveis, os atributos de momentum de 252 e 126 pregões possuem as maiores participações médias, '
        'seguidos por medidas de risco de mercado, volatilidade e tendência de mercado de 63 pregões. '
        'A presença dessas variáveis descreve os modelos salvos; não prova que elas tenham acrescentado poder preditivo fora da amostra.',
        'Para cada ativo, normaliza-se a média do valor absoluto de SHAP pela soma de seus atributos; '
        'as participações são então promediadas igualmente entre os vinte ativos. '
        'Valores absolutos informam intensidade de contribuição, sem indicar se um atributo eleva ou reduz a previsão. '
        'Os arquivos disponíveis são resumos dos modelos salvos, e não explicações individuais das previsões OOF de 2025. '
        'Portanto, a figura não demonstra estabilidade temporal, causalidade econômica ou importância exclusiva no teste.',
        'O conjunto de evidências aponta desempenho heterogêneo e capacidade limitada de previsão de magnitude. '
        'O retorno da carteira não constitui validação isolada do XGBoost. '
        'Embora o horizonte tenha sido escolhido em 2024, os artefatos preservam a seleção original entre regressor e classificador, '
        'que pode consultar o OOF completo. A validação interna de hiperparâmetros também precisa ser auditada quanto à separação de alvos sobrepostos. '
        'Uma avaliação confirmatória exige reconstruir treino e seleção de forma aninhada antes de um novo período de teste.']
    fragment='\n\n'.join(text)+'\n'
    (DEST/'predicao_xgb.tex').write_text(fragment,encoding='utf-8')
    target=TCC/'05_Resultados/predicao_xgb'
    target.mkdir(parents=True,exist_ok=True)
    for p in DEST.glob('*.png'): shutil.copyfile(p,target/p.name)
    (TCC/'05_Resultados/predicao_xgb.tex').write_text(fragment,encoding='utf-8')
    chapter=TCC/'05_Resultados/resultados.tex'
    current=chapter.read_text(encoding='utf-8')
    include=r'\input{05_Resultados/predicao_xgb.tex}'
    marker=r'\section{Comparação com ativos comuns}'
    if include not in current:
        if marker not in current: raise ValueError('Ponto de insercao do capitulo nao encontrado.')
        current=current.replace(marker,include+'\n\n'+marker)
    chapter.write_text(current,encoding='utf-8')
    print(json.dumps(summary,ensure_ascii=False,indent=2))


def figure(name,caption):
    return '\n'.join([r'\begin{figure}[H]\centering', '\\caption{'+caption+'}',
        r'\includegraphics[width=0.92\textwidth]{05_Resultados/predicao_xgb/'+name+'}',
        r'\par\smallskip{\footnotesize Fonte: elaboração própria.}\end{figure}'])


if __name__=='__main__':
    main()
