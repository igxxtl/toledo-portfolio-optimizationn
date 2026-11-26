"""Relatorio da rodada experimental, sem alterar modelos atuais ou o TCC."""
from pathlib import Path
import json
import os
import numpy as np
import pandas as pd

ROOT=Path(__file__).resolve().parents[1]
OUT=ROOT/'analise_resultados/melhoria_xgb'


def main():
    os.environ.setdefault('MPLCONFIGDIR',str(ROOT/'tmp/xgb_matplotlib_cache'))
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    summary=pd.read_csv(OUT/'resumo_horizontes.csv')
    comparison=pd.read_csv(OUT/'comparacao_pareada_2025.csv')
    selected=pd.read_csv(OUT/'comparacao_horizontes_atuais.csv')
    scores=pd.read_csv(OUT/'metricas_todas.csv')
    choices=pd.read_csv(OUT/'escolhas_validacao_2024.csv')
    result=json.loads((OUT/'resultado.json').read_text(encoding='utf-8'))
    protocol=json.loads((OUT/'protocolo.json').read_text(encoding='utf-8'))
    audit=pd.read_csv(OUT/'auditoria_temporal.csv')
    assert (pd.to_datetime(audit.max_train_maturity)<pd.to_datetime(audit.first_prediction)).all()
    assert (pd.to_datetime(audit.max_fit_maturity)<pd.to_datetime(audit.early_stop_start)).all()
    assert len(comparison)==60 and len(selected)==20
    effects=[]
    tests=scores[scores.phase.eq('test')]
    # Contrastes dentro do protocolo novo: variam pesos/perfil/perda, nao calendario.
    contrasts=[('sq_regular_w5','sq_regular_w0','peso_5_para_0'),
               ('sq_regular_w5','sq_regular_w1','peso_5_para_1'),
               ('sq_regular_w0','sq_conservador_w0','regularizacao_peso_0'),
               ('sq_regular_w1','sq_conservador_w1','regularizacao_peso_1'),
               ('sq_regular_w5','sq_conservador_w5','regularizacao_peso_5'),
               ('sq_conservador_w0','abs_conservador_w0','perda_absoluta_peso_0'),
               ('sq_conservador_w1','abs_conservador_w1','perda_absoluta_peso_1')]
    for before,after,name in contrasts:
        a=tests[tests.model.eq(before)].set_index(['ticker','horizon'])
        b=tests[tests.model.eq(after)].set_index(['ticker','horizon'])
        for (ticker,h),r in a.iterrows():
            other=b.loc[(ticker,h)]
            effects.append(dict(contrast=name,ticker=ticker,horizon=h,
                rmse_reduction_pct=(1-other.rmse/r.rmse)*100,
                mae_reduction_pct=(1-other.mae/r.mae)*100,hr_gain_pp=other.hr-r.hr))
    effects=pd.DataFrame(effects);effects.to_csv(OUT/'efeitos_controlados.csv',index=False)
    effects.groupby(['contrast','horizon'],as_index=False).agg(
        assets=('ticker','size'),median_rmse_reduction_pct=('rmse_reduction_pct','median'),
        mean_rmse_reduction_pct=('rmse_reduction_pct','mean'),
        wins_rmse=('rmse_reduction_pct',lambda s:int((s>0).sum()))).to_csv(OUT/'efeitos_controlados_resumo.csv',index=False)
    # Comparacoes trimestrais em pares identicos; selecao permanece aquela de 2024.
    obs=pd.read_csv(OUT/'previsoes.csv')
    quarter=[];chosen_observations=[];amplitudes=[]
    for c in choices.itertuples():
        g=obs[(obs.ticker==c.ticker)&(obs.horizon==c.horizon)&obs.phase.eq('test')]
        chosen_observations.append(g[['ticker','horizon','date','maturity','fold','actual','current',c.chosen]].rename(
            columns={c.chosen:'candidate'}).assign(chosen=c.chosen))
        amplitudes.append(dict(ticker=c.ticker,horizon=c.horizon,chosen=c.chosen,
            std_current=float(g.current.std(ddof=0)),std_candidate=float(g[c.chosen].std(ddof=0)),
            std_actual=float(g.actual.std(ddof=0)),correlation_current=float(g.current.corr(g.actual)),
            correlation_candidate=float(g[c.chosen].corr(g.actual))))
        for fold,f in g.groupby('fold'):
            ec=float(np.sqrt(np.mean((f.current-f.actual)**2)))
            en=float(np.sqrt(np.mean((f[c.chosen]-f.actual)**2)))
            quarter.append(dict(ticker=c.ticker,horizon=c.horizon,fold=fold,chosen=c.chosen,n=len(f),
                                rmse_current=ec,rmse_new=en,rmse_reduction_pct=(1-en/ec)*100))
    pd.DataFrame(quarter).to_csv(OUT/'comparacao_trimestres.csv',index=False)
    pd.concat(chosen_observations).to_csv(OUT/'previsoes_escolhidas_2025.csv',index=False)
    pd.DataFrame(amplitudes).to_csv(OUT/'amplitude_e_correlacao.csv',index=False)
    plt.rcParams.update({'font.family':'DejaVu Sans','font.size':10,'axes.spines.top':False,'axes.spines.right':False})
    fig,axes=plt.subplots(1,2,figsize=(10,4.8))
    for ax,metric in zip(axes,['mae','rmse']):
        x=np.arange(len(summary));width=.35
        ax.bar(x-width/2,summary[metric+'_current']*100,width,label='Atual',color='#8a9ba8')
        ax.bar(x+width/2,summary[metric+'_new']*100,width,label='Candidato escolhido em 2024',color='#227d70')
        ax.set_xticks(x,[str(int(h))+' pregões' for h in summary.horizon])
        ax.set_ylabel(metric.upper()+' × 100 (unidades de log-retorno)');ax.legend(fontsize=8)
        ax.grid(axis='y',alpha=.15)
    fig.suptitle('Comparação pareada em 2025 · médias entre os 20 ativos\nComparar atual e candidato dentro de cada horizonte')
    fig.tight_layout();fig.savefig(OUT/'comparacao_erros.png',dpi=180);plt.close(fig)
    pivot=comparison.pivot(index='ticker',columns='horizon',values='rmse_reduction_pct')
    fig,ax=plt.subplots(figsize=(7,8))
    limit=max(float(abs(pivot.to_numpy()).max()),1.)
    im=ax.imshow(pivot,cmap='RdBu',vmin=-limit,vmax=limit,aspect='auto')
    ax.set_yticks(range(len(pivot)),pivot.index);ax.set_xticks(range(3),['5 pregões','10 pregões','21 pregões'])
    for i in range(len(pivot)):
        for j in range(3):
            val=pivot.iloc[i,j];ax.text(j,i,f'{val:.1f}',ha='center',va='center',fontsize=8,
                color='white' if abs(val)>.55*limit else '#182b35')
    ax.set_title('Redução do RMSE contra a previsão atual (%)\nPositivo: melhora · Negativo: piora')
    fig.colorbar(im,ax=ax,label='Redução relativa do RMSE (%)');fig.tight_layout()
    fig.savefig(OUT/'melhoria_por_ativo.png',dpi=180);plt.close(fig)
    fmt=lambda v:f'{v:.2f}'.replace('.',',')
    lines=['TESTES DE MELHORIA DO XGBOOST — COMPARAÇÃO COM OS RESULTADOS ATUAIS','',
        'Os modelos e previsões atuais foram preservados. Foram testadas oito configurações de regressão pura: '
        'perda quadrática com dois perfis de complexidade e pesos temporais lambda=0/1/5; '
        'perda absoluta com perfil conservador e lambda=0/1.',
        'A configuração de cada ativo/horizonte foi escolhida pelo menor RMSE em 2024. '
        'Essa escolha ficou fixa para a comparação de 2025. Os oito candidatos fixos também são apresentados, '
        'mas seu ranking em 2025 não foi usado para escolher o candidato.',
        'Treinos expansivos com alvos maturados antes de cada fold. Purga adicional entre treino e early stopping. '
        'Após escolher o número de árvores com a validação interna, refit em todos os alvos conhecidos. '
        'Em 2025, atualização trimestral, sem reajuste de hiperparâmetros pelo resultado do teste.',
        'O perfil conservador altera profundidade, peso mínimo da folha e regularização conjuntamente. '
        'A comparação entre perfis não identifica isoladamente cada hiperparâmetro. '
        'O candidato abs busca erro absoluto, mas ainda é selecionado por RMSE para manter a pergunta sobre magnitude.',
        'Erros abaixo são médias simples entre os vinte ativos, na escala de log-retorno multiplicada por 100. '
        'Não comparar níveis de erro de horizontes diferentes como se fossem o mesmo alvo.','']
    for r in summary.itertuples():
        lines += [f'{int(r.horizon)} pregões:',
            f'  MAE atual {fmt(r.mae_current*100)} → candidato {fmt(r.mae_new*100)}.',
            f'  RMSE atual {fmt(r.rmse_current*100)} → candidato {fmt(r.rmse_new*100)}; redução {fmt(r.rmse_reduction_pct)}%.',
            f'  Acerto direcional atual {fmt(r.hr_current)}% → candidato {fmt(r.hr_new)}%.',
            f'  Melhora de RMSE em {int(r.wins_rmse)}/20 ativos; supera retorno zero em {int(r.beat_zero)}/20.','']
    c=result['current_horizon']
    lines += ['Mantendo o horizonte atualmente utilizado por cada ativo:',
        f'  RMSE médio atual {fmt(c["rmse_current"]*100)} → candidato {fmt(c["rmse_new"]*100)}.',
        f'  MAE médio atual {fmt(c["mae_current"]*100)} → candidato {fmt(c["mae_new"]*100)}.',
        f'  Melhora de RMSE em {c["wins_rmse"]}/20 ativos.',
        f'  Redução do RMSE médio: {fmt((1-c["rmse_new"]/c["rmse_current"])*100)}%; '
        f'acurácia média: {fmt(c["hr_current"])}% → {fmt(c["hr_new"])}%.',
        f'No conjunto das 60 combinações, melhora frente ao atual em {result["wins_vs_current"]}/60, '
        f'frente ao retorno zero em {result["wins_vs_zero"]}/60 e frente à média histórica em {result["wins_vs_historical"]}/60.','',
        'INTERPRETAÇÃO E LIMITES',
        'Nesta execução, as pequenas reduções de erro em cinco e 21 pregões não resolvem a baixa precisão de magnitude. '
        'No horizonte de 21 pregões, o acerto de direção também cai. Mantendo os horizontes usados atualmente no sistema, '
        'o erro médio praticamente empata com o atual e só metade dos ativos melhora. '
        'A rodada não sustenta substituição global do pipeline apenas por esses resultados.',
        'Melhorar contra o modelo antigo não basta: o candidato também deve superar referências simples. '
        'Prever perto de zero pode reduzir RMSE sem revelar sinal útil. '
        'Resultados positivos em apenas alguns ativos devem ser tratados como heterogeneidade, não como justificativa para seleção retrospectiva.',
        '2025 já havia sido examinado nesta pesquisa. A rodada é comparativa retrospectiva, não um novo teste independente. '
        'Os hiperparâmetros foram escolhidos usando apenas a validação anterior, mas o desenho experimental foi motivado pelos diagnósticos observados. '
        'O histórico de preços locais e a seleção fixa das vinte ações mantêm as limitações de sobrevivência e revisões dos dados.',
        'O baseline é o conjunto de previsões originalmente salvo, inclusive a seleção histórica entre regressor e classificador. '
        'As candidatas usam um novo protocolo de treino; a diferença contra o baseline não isola uma única mudança. '
        'Os contrastes em efeitos_controlados.csv comparam pesos/perfis/perdas dentro do mesmo protocolo novo.',
        'Não foram realizados testes de significância que suponham observações independentes, nem atualização do BL, backtest financeiro ou substituição automática do pipeline.',
        'Versões: '+json.dumps(protocol['versions'],ensure_ascii=False),
        'Verificações: maturidade antes de cada decisão; purga no early stopping; mesmos pares; hashes dos artefatos originais inalterados.']
    (OUT/'RELATORIO.txt').write_text('\n'.join(lines)+'\n',encoding='utf-8')
    print('Relatorio e duas figuras gerados; efeitos controlados e trimestres exportados.')


if __name__=='__main__':main()
