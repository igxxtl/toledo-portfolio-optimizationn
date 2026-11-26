"""Relatorio financeiro comparativo da substituicao do XGB individual pelo pool."""
from pathlib import Path
import json
import os
import numpy as np
import pandas as pd

ROOT=Path(__file__).resolve().parents[1]
OUT=ROOT/'analise_resultados/simulacao_pool'
NAMES={'BL_atual':'BL atual','BL_pool':'BL com pool','BL_pool_omega_fixo':'BL pool / Ω atual',
       'XGB_atual':'XGB atual direto','XGB_pool':'Pool direto','Prior':'Prior sem views','Markowitz':'Markowitz'}


def main():
    result=pd.read_csv(OUT/'metricas.csv');summary=pd.read_csv(OUT/'resumo.csv')
    paired=pd.read_csv(OUT/'comparacao_pareada.csv');compare=pd.read_csv(OUT/'comparacao_resumo.csv')
    curve=pd.read_csv(OUT/'series_selecao_propria.csv',parse_dates=['date'])
    audit=pd.read_csv(OUT/'auditoria_comparada.csv')
    verify=json.loads((OUT/'verificacao.json').read_text(encoding='utf-8'))
    assert verify['status']=='passed' and verify['runs']==len(result)
    free=result[result.scenario.eq('escolha_livre')]
    free.to_csv(OUT/'resultados_selecao_propria.csv',index=False)
    free[free.n.isin([5,10,20])].to_csv(OUT/'comparacao_5_10_20.csv',index=False)
    all20=free[free.n.eq(20)].set_index('model')
    metrics=['return_pct','sharpe','max_drawdown_pct','turnover_total','cost_brl','final_capital_brl']
    differences=pd.DataFrame({c:all20.loc[c,metrics]-all20.loc['BL_atual',metrics] for c in ['BL_pool','BL_pool_omega_fixo','XGB_pool','Prior']}).T
    differences.index.name='model';differences.to_csv(OUT/'diferencas_n20.csv')
    (OUT/'resultado.json').write_text(json.dumps(dict(
        runs=len(result),bl_new_return_n20=float(all20.loc['BL_pool','return_pct']),
        bl_old_return_n20=float(all20.loc['BL_atual','return_pct']),
        bl_return_delta_n20_pp=float(all20.loc['BL_pool','return_pct']-all20.loc['BL_atual','return_pct']),
        direct_old_return_n20=float(all20.loc['XGB_atual','return_pct']),
        direct_pool_return_n20=float(all20.loc['XGB_pool','return_pct']),
        bl_new_sharpe_n20=float(all20.loc['BL_pool','sharpe']),
        bl_old_sharpe_n20=float(all20.loc['BL_atual','sharpe']),
        gate_pairs=int(audit.news_count.ge(3).sum()),baseline_reproduced=verify['original_runs_reproduced']),indent=2),encoding='utf-8')
    os.environ.setdefault('MPLCONFIGDIR',str(ROOT/'tmp/xgb_matplotlib_cache'))
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    plt.rcParams.update({'font.family':'DejaVu Sans','font.size':10,'axes.spines.top':False,'axes.spines.right':False})
    colors={'BL_atual':'#2468a0','BL_pool':'#168465','BL_pool_omega_fixo':'#75aa85','Prior':'#888888','Markowitz':'#c7742e',
            'XGB_atual':'#9b4f5c','XGB_pool':'#634eaa'}
    fig,axes=plt.subplots(1,2,figsize=(12,5))
    for ax,arms in [(axes[0],['BL_atual','BL_pool','BL_pool_omega_fixo','Prior','Markowitz']),
                    (axes[1],['XGB_atual','XGB_pool'])]:
        for arm in arms:
            g=free[free.model.eq(arm)].sort_values('n')
            ax.plot(g.n,g.return_pct,label=NAMES[arm],color=colors[arm],lw=2,ls='--' if arm=='Prior' else '-')
        ax.axhline(float(free.selic_return_pct.iloc[0]),color='#444',ls=':',label='Selic hipotética')
        ax.set_xlabel('Número de ativos');ax.set_ylabel('Retorno líquido (%)');ax.set_xticks([1,5,10,15,20]);ax.grid(alpha=.15);ax.legend(fontsize=8)
    fig.suptitle('Comparação da nova configuração · seleção própria em 2025\nMesmos horizontes, custos de 10 bps, restrições e rebalanceamento mensal')
    fig.tight_layout();fig.savefig(OUT/'retornos_cardinalidade.png',dpi=180);plt.close(fig)
    fig,axes=plt.subplots(1,3,figsize=(13,4.8))
    rf=(1.15)**(1/252)-1
    for ax,n in zip(axes,[5,10,20]):
        for arm in ['BL_atual','BL_pool','Prior','Markowitz']:
            g=curve[curve.n.eq(n)&curve.model.eq(arm)].sort_values('date')
            ax.plot(g.date,g.capital/1000,label=NAMES[arm],color=colors[arm],lw=1.7,ls='--' if arm=='Prior' else '-')
        dates=curve.date.drop_duplicates().sort_values()
        ax.plot(dates,100*(1+rf)**np.arange(1,len(dates)+1),color='#444',ls=':',label='Selic hipotética')
        ax.set_title(f'n = {n}');ax.set_ylabel('Capital (R$ mil)');ax.tick_params(axis='x',rotation=30);ax.grid(alpha=.15)
    axes[0].legend(fontsize=8);fig.suptitle('Capital líquido · seleção própria · comparação retrospectiva de 2025')
    fig.tight_layout();fig.savefig(OUT/'trajetorias_comparadas.png',dpi=180);plt.close(fig)
    common=paired[paired.scenario.eq('mesmos_ativos')]
    band=common.groupby('n').BL_pool_minus_current_pp.agg(median='median',low=lambda s:s.quantile(.1),high=lambda s:s.quantile(.9)).reset_index()
    fig,ax=plt.subplots(figsize=(8,4.8))
    ax.fill_between(band.n.to_numpy(),band.low.to_numpy(),band.high.to_numpy(),color='#168465',alpha=.18,label='P10–P90 entre subconjuntos')
    ax.plot(band.n,band['median'],color='#168465',marker='o',label='Diferença pareada mediana')
    ax.axhline(0,color='#555',ls='--',lw=1);ax.set_xticks([1,5,10,15,20]);ax.set_xlabel('Número de ativos')
    ax.set_ylabel('Retorno BL pool − BL atual (p.p.)');ax.legend(fontsize=9);ax.grid(alpha=.15)
    ax.set_title('Mesmos ativos · efeito da nova configuração na alocação\nFaixa é dispersão de composições, não intervalo de confiança')
    fig.tight_layout();fig.savefig(OUT/'ganho_pareado.png',dpi=180);plt.close(fig)
    fmt=lambda v:f'{v:.2f}'.replace('.',',')
    lines=['SIMULAÇÃO FINANCEIRA DA NOVA CONFIGURAÇÃO XGBOOST CONJUNTA','',
        'PROTOCOLO',
        'Pool fixo das vinte ações, não a receita mista escolhida por método em 2024. A arquitetura, os grupos de features '
        'e os parâmetros são os mesmos da rodada preditiva anterior. Não foi realizada otimização para estes retornos.',
        'Período: 02/01/2025 a 30/12/2025; capital inicial R$ 100 mil; 250 pregões; custos de 10 bps; '
        'sinal no fechamento e execução no fechamento seguinte; mesmos limites, piso, covariância, aversão ao risco e frequência mensal.',
        'Mantidos os horizontes escolhidos em 2024 para cada ativo. São 551 subconjuntos comuns (semente 2025), '
        'mais vinte trajetórias com seleção própria, por configuração. Sete configurações, totalizando '+str(len(result))+' simulações.',
        'BL atual: pipeline original. BL pool: nova previsão e confiança recalculada pelos próprios erros já maturados. '
        'BL pool/Omega atual: nova previsão e a matriz de incerteza original, para distinguir mudança da opinião e da confiança. '
        'XGB direto e pool direto: expectativa quantitativa entra no mesmo otimizador de média-variância, sem atualização BL. '
        'Prior sem views e Markowitz são referências.',
        'Os 1.142 resultados de BL e Markowitz da rodada original foram reproduzidos numericamente. '
        'Arquivos originais e parâmetros foram preservados.',
        'Foram exportadas novamente todas as previsões necessárias, inclusive aquelas de dezembro cujo desfecho é posterior ao fim do teste. '
        'O treino não usa esses desfechos; a confiança só consulta erros com maturidade conhecida no instante de cada sinal. '
        'Todas as previsões já avaliadas anteriormente foram reproduzidas, sem preenchimento arbitrário.','',
        'RESULTADOS COM TODO O UNIVERSO (n=20)',
        'Configuração | Retorno líquido | Sharpe | MDD | Capital final']
    for arm in NAMES:
        r=all20.loc[arm]
        lines.append(f'{NAMES[arm]} | {fmt(r.return_pct)}% | {fmt(r.sharpe)} | {fmt(r.max_drawdown_pct)}% | R$ {fmt(r.final_capital_brl)}')
    delta=float(all20.loc['BL_pool','return_pct']-all20.loc['BL_atual','return_pct'])
    lines+=['',f'Diferença BL pool menos BL atual: {fmt(delta)} pontos percentuais.',
        'Com vinte ativos, a elegibilidade é a mesma e o contraste se concentra na distribuição de pesos. '
        'Nos demais tamanhos com seleção própria, as mudanças incluem seleção e alocação.','',
        'SELEÇÃO PRÓPRIA — ILUSTRAÇÕES EM n=5 E n=10']
    for n in [5,10]:
        g=free[free.n.eq(n)].set_index('model')
        lines.append(f'n={n}: BL atual {fmt(g.loc["BL_atual","return_pct"])}%; BL pool {fmt(g.loc["BL_pool","return_pct"])}%; '
            f'controle Omega atual {fmt(g.loc["BL_pool_omega_fixo","return_pct"])}%; '
            f'XGB direto {fmt(g.loc["XGB_atual","return_pct"])}%; pool direto {fmt(g.loc["XGB_pool","return_pct"])}%.')
    lines+=['','MESMOS SUBCONJUNTOS — COMPARAÇÃO PAREADA']
    for n in [5,10,20]:
        r=compare[compare.scenario.eq('mesmos_ativos')&compare.n.eq(n)].iloc[0]
        lines.append(f'n={n}: diferença mediana BL pool − atual {fmt(r.median_BL_gain_pp)} p.p.; '
            f'pool vence em {fmt(r.wins_BL_pct)}% e empata em {fmt(r.ties_BL_pct)}% dos subconjuntos avaliados.')
    lines+=['','INTERPRETAÇÃO E LIMITES',
        'O resultado financeiro do BL é misto: com seleção própria, melhora em alguns tamanhos e piora em outros. '
        'O ganho em dez ativos não autoriza concluir que a configuração nova seja superior para qualquer carteira.',
        f'Com vinte ativos, o Sharpe do BL passa de {fmt(all20.loc["BL_atual","sharpe"])} para '
        f'{fmt(all20.loc["BL_pool","sharpe"])}. No XGB direto, o retorno melhora, mas a perda máxima '
        f'passa de {fmt(all20.loc["XGB_atual","max_drawdown_pct"])}% para '
        f'{fmt(all20.loc["XGB_pool","max_drawdown_pct"])}%; o pool direto permanece abaixo '
        f'da referência Selic de {fmt(float(free.selic_return_pct.iloc[0]))}% nesse tamanho.',
        'O controle que mantém Omega praticamente reproduz o BL pool nos três tamanhos ilustrativos; '
        'a mudança da confiança tem efeito financeiro pequeno nessa comparação.',
        'A redução de erro preditivo da rodada anterior não garante aumento de retorno financeiro. '
        'O gate de três manchetes habilita apenas seis de 240 pares ativo-data. '
        'Nos demais, o posterior fica próximo do prior, embora a covariância possa transmitir a influência de views de outros ativos. '
        'Essa regra pode limitar a influência da nova previsão.',
        'O controle com Omega fixo não é o novo sistema completo: ele isola a mudança das opiniões mantendo a confiança anterior. '
        'O BL pool principal atualiza a confiança com os próprios erros históricos conhecidos, sem usar o resultado da previsão em curso.',
        'A escolha de destacar o pool foi feita após examinar seus resultados preditivos em 2025. '
        'Esta simulação é retrospectiva e não é confirmação independente. Não houve novos dados nem seleção de melhor cardinalidade para promover o modelo. '
        'Não interpretar 3.997 simulações como amostras independentes, nem percentis entre subconjuntos como intervalos de confiança.',
        'Resultados e figuras foram guardados separadamente; o modelo atual e o TCC não foram substituídos automaticamente.']
    (OUT/'RELATORIO.txt').write_text('\n'.join(lines)+'\n',encoding='utf-8')
    print(json.dumps(json.loads((OUT/'resultado.json').read_text()),ensure_ascii=False,indent=2))


if __name__=='__main__':main()
