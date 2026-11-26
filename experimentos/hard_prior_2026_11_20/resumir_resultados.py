from pathlib import Path
import os
import sys
import json

import numpy as np
import pandas as pd

OUT = Path(__file__).resolve().parent
os.environ['MPLCONFIGDIR'] = str(OUT/'cache_matplotlib')
sys.path.append(r'C:\Users\igor.toledo_visagio\Documents\GitHub\toledo-portfolio-optimizationn\tmp\xgb_plot_deps')
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

metrics = pd.read_csv(OUT/'metricas.csv')
summary = pd.read_csv(OUT/'resumo.csv')
paired = pd.read_csv(OUT/'comparacao_pareada.csv')
curves = pd.read_csv(OUT/'series_selecao_propria.csv',parse_dates=['date'])
trades = pd.read_csv(OUT/'rebalanceamentos.csv')
verification = json.loads((OUT/'verificacao.json').read_text(encoding='utf-8'))
assert verification['status']=='passed'
assert np.isfinite(metrics.select_dtypes('number').to_numpy()).all()
for row in trades.itertuples():
    w = np.array(list(json.loads(row.weights).values()))
    assert np.isclose(w.sum(),1,atol=1e-8)
    assert w.min() >= .001-1e-8
    assert w.max() <= row.cap+1e-8
    assert pd.Timestamp(row.execution_date)>pd.Timestamp(row.signal_date)
names = {'atual':'Atual: hard prior + multiplicador',
         'sem_hard_prior':'Sem hard prior; mantém multiplicador',
         'somente_gate':'Somente gate; sem multiplicador',
         'Prior':'Prior sem opiniões', 'Markowitz':'Markowitz'}
colors = {'atual':'#49687f','sem_hard_prior':'#dc7633','somente_gate':'#198377',
          'Prior':'#999999','Markowitz':'#a364a4'}
fig, axes = plt.subplots(1,2,figsize=(13,5.2))
for name in names:
    g=curves[curves.n.eq(20)&curves.model.eq(name)].sort_values('date')
    axes[0].plot(g.date,g.capital/1000,label=names[name],color=colors[name],
                 linestyle='--' if name=='Prior' else '-',linewidth=1.9)
axes[0].set(title='Carteira com as 20 ações',ylabel='Patrimônio líquido (R$ mil)')
axes[0].tick_params(axis='x',rotation=25)
for name in ['atual','sem_hard_prior','somente_gate']:
    g=summary[summary.scenario.eq('mesmos_ativos')&summary.model.eq(name)].sort_values('n')
    axes[1].plot(g.n,g.return_median,color=colors[name],label=names[name],linewidth=2)
axes[1].set(title='Mediana dos conjuntos iguais de ativos',xlabel='Quantidade de ações',ylabel='Retorno líquido em 2025 (%)',xticks=[1,5,10,15,20])
for ax in axes:
    ax.grid(alpha=.22)
    ax.spines[['top','right']].set_visible(False)
handles,labels=axes[0].get_legend_handles_labels()
fig.legend(handles,labels,loc='lower center',ncol=2,fontsize=9,frameon=False)
fig.suptitle('Remoção do hard prior — comparação retrospectiva, parâmetros preservados',fontsize=13)
fig.tight_layout(rect=[0,.17,1,.94])
fig.savefig(OUT/'comparacao_hard_prior.png',dpi=180,bbox_inches='tight')
plt.close(fig)
full=metrics[metrics.scenario.eq('escolha_livre')&metrics.n.eq(20)]
parts=['TESTE DE REMOCAO DO HARD PRIOR — 2025',
       'XGBoost conjunto (pool), previsoes ja existentes; sem treinamento ou recalibracao.',
       'Gate=3; rebalanceamento mensal; custos=0,1% por turnover; horizonte por ativo preservado.',
       'Mesmo motor, noticias, datas, covariancias, restricoes e conjuntos de ativos.',
       'Configuracao atual preservada em disco. Saidas historicas nao alteradas.',
       '', 'CARTEIRA COM AS 20 ACOES',
       full[['model','return_pct','sharpe','max_drawdown_pct','cost_brl','turnover_total']].to_string(index=False),
       '', 'COMPARACOES PAREADAS COM OS MESMOS ATIVOS']
pair_stats=[]
for name in ['sem_hard_prior','somente_gate']:
    p=paired[paired.scenario.eq('mesmos_ativos')]
    delta=p[name+'_delta_pp']
    row=dict(model=name,pairs=len(p),wins=int((delta>1e-8).sum()),
             ties=int((delta.abs()<=1e-8).sum()),losses=int((delta < -1e-8).sum()),
             median_gain_pp=float(delta.median()))
    pair_stats.append(row)
    parts.append(json.dumps(row,ensure_ascii=False))
parts += ['', 'MEDIANA DOS RETORNOS POR QUANTIDADE DE ATIVOS',
          summary[summary.scenario.eq('mesmos_ativos')].pivot(index='n',columns='model',values='return_median').to_string(),
          '', 'AUDITORIA E LIMITACOES',
          f"Baseline reproduzido em {verification['baseline_reproduced']} carteiras. Q igual entre as alternativas.",
          'Nos 6 pares ativo-data com pelo menos 3 noticias, Omega permanece igual.',
          'Nos outros 234 pares, a retirada do hard prior aumenta a participacao das opinioes.',
          'A retirada ativa tambem as previsoes quantitativas; eventual ganho nao demonstra valor isolado das noticias.',
          'Sensibilidade retrospectiva em 2025; nao representa confirmacao em periodo novo.',
          'Sharpe considera a referencia livre de risco de 15% ao ano; retorno ja desconta custos.',
          'Verificados pesos positivos, soma=1, limites de concentracao e execucao apos o sinal.']
(OUT/'RELATORIO.txt').write_text('\n'.join(parts),encoding='utf-8')
pd.DataFrame(pair_stats).to_csv(OUT/'ganhos_pareados.csv',index=False)
full.to_csv(OUT/'comparacao_20_acoes.csv',index=False)
print('\n'.join(parts[:12]))
