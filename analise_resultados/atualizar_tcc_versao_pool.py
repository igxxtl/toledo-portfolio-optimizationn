"""Apresenta somente a configuracao pool no TCC, preservando as referencias financeiras."""
from pathlib import Path
from datetime import datetime
from zipfile import ZipFile, ZIP_DEFLATED
import difflib
import hashlib
import json
import os
import re
import numpy as np
import pandas as pd

REPO=Path(__file__).resolve().parents[1]
TCC=Path(r'C:/Users/igor.toledo_visagio/Documents/TCC 2')
DATA=REPO/'analise_resultados/simulacao_pool'
OUT=REPO/'output'/('tcc_versao_pool_'+datetime.now().strftime('%Y%m%d_%H%M%S'))
OUT.mkdir(parents=True,exist_ok=True)
original={p:p.read_bytes() for p in TCC.rglob('*') if p.is_file()}
texts={p:v.decode('utf-8-sig').replace('\r\n','\n') for p,v in original.items() if p.suffix=='.tex'}
with ZipFile(OUT/'backup_anterior.zip','w',ZIP_DEFLATED) as z:
    for p,v in original.items():z.writestr(p.relative_to(TCC).as_posix(),v)

def f(v,n=2):return f'{v:,.{n}f}'.replace(',','X').replace('.',',').replace('X','.')
def row(values):return ' & '.join(map(str,values))+r' \\'+'\n'
def table(caption,label,columns,header,body,source):
    return ('\\begin{table}[H]\n\\centering\n\\caption{'+caption+'}\n\\label{'+label+'}\n{\\footnotesize\n'
        '\\begin{tabular}{@{}'+columns+'@{}}\n\\hline\n'+header+r' \\ \hline'+'\n'+body+
        '\\hline\n\\end{tabular}}\n\\par\\smallskip{\\footnotesize Fonte: elaboração própria. '+source+'}\n\\end{table}\n')
def once(rel,old,new):
    p=TCC/rel
    assert texts[p].count(old)==1,(rel,old[:90],texts[p].count(old))
    texts[p]=texts[p].replace(old,new,1)
def block(rel,start,end,new):
    p=TCC/rel;a=texts[p].index(start);b=texts[p].index(end,a)
    texts[p]=texts[p][:a]+new+texts[p][b:]

# Somente as quatro configuracoes pertinentes a versao final.
metrics=pd.read_csv(DATA/'metricas.csv')
arms=['BL_pool','Markowitz','Prior','XGB_pool']
m=metrics[metrics.model.isin(arms)].copy()
assert len(m)==2284
free=m[m.scenario.eq('escolha_livre')]
common=m[m.scenario.eq('mesmos_ativos')]
gfree={n:g.set_index('model') for n,g in free.groupby('n')}
med=common.groupby(['n','model']).median(numeric_only=True)
pair=common.pivot(index=['n','repetition'],columns='model',values='return_pct')
pair['delta']=pair.BL_pool-pair.Markowitz
paired=pair.groupby('n').agg(count=('delta','size'),delta=('delta','median'),wins=('delta',lambda x:float((x>1e-8).mean()*100)))
q=common[common.model.eq('BL_pool')].groupby('n').return_pct.quantile([.1,.9]).unstack()
curve=pd.read_csv(DATA/'series_selecao_propria.csv',parse_dates=['date'])
curve=curve[curve.model.isin(arms)].copy()
costs=[]
for (n,model),c in curve.groupby(['n','model']):
    c=c.sort_values('date');previous=c.capital.shift(1,fill_value=100000.)
    gross=float((np.prod((c.capital+c.cost_brl)/previous)-1)*100)
    net=float((c.capital.iloc[-1]/100000-1)*100)
    costs.append(dict(n=n,model=model,gross=gross,net=net,impact=gross-net,cost=float(c.cost_brl.sum())))
cost=pd.DataFrame(costs).set_index(['n','model'])
pred=pd.read_csv(REPO/'analise_resultados/metodologia_xgb/previsoes.csv',parse_dates=['date','maturity'])
pred=pred[['ticker','horizon','date','maturity','phase','actual','pool']]
choices=pd.read_csv(DATA/'horizontes_mantidos.csv')
choices['ticker']=choices.asset.str.replace('.SA','',regex=False)
chosen=pred.merge(choices[['ticker','horizon']],on=['ticker','horizon'],validate='many_to_one')
score=[]
for (ticker,h,phase),p in chosen.groupby(['ticker','horizon','phase']):
    y=p.actual.to_numpy();v=p.pool.to_numpy();err=y-v
    score.append(dict(ticker=ticker,horizon=int(h),phase=phase,n=len(p),
        hr=float(np.mean(np.sign(y)==np.sign(v))*100),mae=float(abs(err).mean()*100),
        rmse=float(np.sqrt(np.mean(err**2))*100),r2=float(1-np.sum(err**2)/np.sum((y-y.mean())**2)),
        majority=float(max(np.mean(y>0),np.mean(y<0),np.mean(y==0))*100)))
scores=pd.DataFrame(score)
test=scores[scores.phase.eq('test')].set_index('ticker')
val=scores[scores.phase.eq('validation')].set_index('ticker')
hr=float(test.hr.mean());r2neg=int((test.r2<0).sum())
beats=int((test.hr>test.majority+1e-8).sum());ties=int(np.isclose(test.hr,test.majority,atol=1e-8,rtol=0).sum())
best=test.hr.idxmax();worst=test.hr.idxmin();positive=test[test.r2>0]
z=gfree[20];wins=int(sum(gfree[n].loc['BL_pool','return_pct']>gfree[n].loc['Markowitz','return_pct'] for n in gfree))
maxret=free[free.model.eq('BL_pool')].sort_values('return_pct').iloc[-1]
maxsh=free[free.model.eq('BL_pool')].sort_values('sharpe').iloc[-1]

rel='05_Resultados/resultados.tex'
prefix=texts[TCC/rel].split('\\section{Desempenho quantitativo')[0]
prefix=prefix.replace('Em complemento, são apresentados os testes de aperfeiçoamento da previsão e a simulação com um modelo conjunto do XGBoost, mantendo a rodada original como referência.',
    'O componente quantitativo utiliza regressão XGBoost conjunta das vinte ações por horizonte, denominada \\emph{pool}. A avaliação inclui suas métricas preditivas e referências que permitem examinar a influência das etapas de alocação.')
prefix=prefix.replace('Total & 1.142 simulações individuais',
    'Comparação principal & 1.142 trajetórias: BL e Markowitz')
prefix=prefix.replace('Estimação e risco &', 'Controles de etapas & 1.142 trajetórias: prior sem views e XGB direto \\\\\nEstimação e risco &')

quant=r'''\section{Desempenho quantitativo e escolha dos horizontes}
As previsões foram produzidas pelo procedimento walk-forward, com regressão conjunta das vinte ações em cada horizonte e identificação do ativo entre as entradas. Os horizontes por ativo seguem a seleção direcional em 2024 previamente fixada no protocolo, aceitando apenas alvos integralmente conhecidos antes de 2025; não foram reeleitos com base no desempenho do pool em 2025. A Tabela~\ref{tab:previsoes_pool_2025} apresenta o diagnóstico do modelo conjunto nos horizontes utilizados na carteira. A coluna de validação descreve o pool em 2024, sem representar uma nova escolha de horizonte pelo maior acerto desse modelo. O número de pares avaliados por ativo em 2025 é 245, 240 ou 229 para horizontes de 5, 10 ou 21 pregões, respectivamente.

'''
quant+=table('Desempenho das previsões utilizadas no experimento.','tab:previsoes_pool_2025','rrrrrr',
    r'Ativo & h & HR val. (\%) & HR 2025 (\%) & Maior classe (\%) & R$^{2}$ 2025',
    ''.join(row([t,int(r.horizon),f(val.loc[t,'hr']),f(r.hr),f(r.majority),f(r.r2,3)]) for t,r in test.iterrows()),
    'HR: acurácia direcional; h: pregões. A maior classe é uma referência descritiva ex post; não é uma regra de negociação escolhida previamente.')
quant+=f'\nA média simples da acurácia entre ativos é {f(hr)}\\% em 2025. O contraste com a frequência da classe majoritária evidencia que acertar mais da metade das observações não assegura superioridade sobre uma referência trivial de sinal. A avaliação financeira exige, portanto, análise própria, sem transformar a acurácia em expectativa direta de retorno.\n\n'
quant+=r'\subsection*{Horizontes utilizados e alcance do sinal preditivo}'+'\n'
quant+=table('Distribuição dos horizontes e diagnóstico agregado.','tab:diagnostico_pool','p{0.47\\linewidth}p{0.43\\linewidth}',
    r'Indicador & Resultado',
    row(['Horizonte de 5 pregões',r'3 ativos (15\%)'])+row(['Horizonte de 10 pregões',r'5 ativos (25\%)'])+
    row(['Horizonte de 21 pregões',r'12 ativos (60\%)'])+row(['Acurácia média em 2025',f(hr)+r'\%'])+
    row([r'R$^{2}$ negativo em 2025',f'{r2neg} dos 20 ativos'])+
    row(['HR acima da classe majoritária',f'{beats} ativo; {ties} empates; {20-beats-ties} abaixo']),
    'Resultados do modelo conjunto, considerando o horizonte utilizado para cada ação.')
quant+=f'''A predominância do horizonte de 21 pregões indica que o protocolo favorece sinais mais longos para a maior parte do universo, sem eliminar heterogeneidade entre empresas. Os extremos do diagnóstico são {best}, com {f(test.loc[best,'hr'])}\\% de acerto direcional, e {worst}, com {f(test.loc[worst,'hr'])}\\%. Entretanto, {best} apenas empata com a frequência da classe majoritária. MGLU3 é o único ativo que supera essa referência descritiva; oito ativos empatam e onze permanecem abaixo dela.

O R$^{{2}}$ avalia a magnitude prevista e pode ser negativo mesmo com acurácia direcional superior a 50\\%. Nos horizontes utilizados, RADL3 é a única exceção positiva, com R$^{{2}}$ de {f(positive.iloc[0].r2,4)}. O resultado dos demais ativos sugere erro quadrático superior ao da média retrospectiva da própria amostra avaliada, sem que essa média constitua uma previsão negociável. Como os alvos de vários pregões se sobrepõem, as observações não são independentes: não se interpreta a diferença para 50\\% como significância estatística automática.

'''
quant+=r'\subsection*{Erros de magnitude e coeficiente de determinação}'+'\n'
ph=pd.read_csv(REPO/'analise_resultados/evolucao_xgb/modelo_conjunto_exploratorio.csv').set_index('horizon')
r2=pd.read_csv(REPO/'analise_resultados/r2_pool/r2_resumo.csv');r2=r2[r2.model.eq('pool')].set_index('horizon')
quant+=table('Métricas do XGBoost conjunto por horizonte em 2025.','tab:metricas_pool_horizontes','rrrrrr',
    r'h & MAE & RMSE & HR (\%) & R$^2$ médio & Ativos com R$^2>0$',
    ''.join(row([h,f(r.mae_pool*100),f(r.rmse_pool*100),f(r.hr_pool),f(r2.loc[h,'r2_mean'],4),int(r2.loc[h,'positive_r2'])]) for h,r in ph.iterrows()),
    'Médias simples entre as vinte ações, nas mesmas datas de cada horizonte. MAE e RMSE expressos em log-retorno multiplicado por 100. O resumo por horizonte difere da composição de horizontes da carteira.')
quant+=r'''No horizonte de 21 pregões, o MAE de 6,72 e o RMSE de 8,42 representam desvios elevados para uma previsão de retorno. Por ser uma média de erros absolutos, o MAE não define uma faixa simétrica garantida em torno de cada previsão. O R$^2$ médio por ativo permanece negativo nos três horizontes. Ao reunir todas as observações antes do cálculo, o R$^2$ é 0,0004, 0,0033 e -0,0085 nos horizontes de 5, 10 e 21 pregões, respectivamente. Essa agregação é distinta da média dos coeficientes individuais e também indica capacidade explicativa próxima de zero.

A conclusão quantitativa é de sinal heterogêneo e fraco na magnitude. Uma eventual vantagem da carteira precisa ser relacionada à combinação de prior, covariância, restrições e custos, e não tomada como prova isolada de qualidade do XGBoost.

'''

same=r'''\section{Comparação com ativos comuns}
A comparação pareada controla a composição elegível: os dois modelos recebem os mesmos papéis e operam nas mesmas datas. A diferença mediana é calculada primeiro dentro de cada par e depois agregada; ela pode diferir da subtração das duas medianas marginais. A Selic acumula 14,87\% nas 250 datas, em vez de 15\%, devido à convenção de 252 dias úteis por ano.

'''
same+=table('Retornos líquidos nos subconjuntos comuns.','tab:retornos_comuns','rrrrrr',
    r'n & Conjuntos & BL med. (\%) & MV med. (\%) & $\Delta$ pareado (p.p.) & BL > MV (\%)',
    ''.join(row([n,int(paired.loc[n,'count']),f(med.loc[(n,'BL_pool'),'return_pct']),f(med.loc[(n,'Markowitz'),'return_pct']),f(paired.loc[n,'delta']),f(paired.loc[n,'wins'],1)]) for n in range(1,21)),
    'MV: Markowitz. A última coluna contabiliza vitórias estritas; os casos n = 1 e n = 2 têm 100\\% de empates.')
same+=f'A vantagem varia com a cardinalidade e a composição. Para n = 5, BL supera Markowitz em {f(paired.loc[5,"wins"],1)}\\% dos conjuntos, com diferença pareada mediana de {f(paired.loc[5,"delta"])} p.p. Para n = 10, a frequência é {f(paired.loc[10,"wins"],1)}\\% e a diferença mediana é {f(paired.loc[10,"delta"])} p.p. Essas frequências descrevem os subconjuntos avaliados, não probabilidades de vitória em outro ano.\n\n'
same+=r'''\subsection*{Dispersão entre conjuntos e risco das carteiras}
\begin{figure}[H]
\centering
\caption{Retorno por cardinalidade com controle dos ativos elegíveis.}
\includegraphics[width=\textwidth]{05_Resultados/figures/retorno liquido ativos.png}
\par\smallskip{\footnotesize Fonte: elaboração própria. Faixas: percentis entre subconjuntos, sem interpretação de intervalo de confiança.}
\end{figure}
'''
same+=table('Risco e desempenho ajustado nos conjuntos comuns.','tab:risco_comuns','rlrrr',
    r'n & Modelo & Sharpe med. & Vol. anual (\%) & MDD med. (\%)',
    ''.join(row([n,name,f(med.loc[(n,arm),'sharpe']),f(med.loc[(n,arm),'volatility_pct']),f(med.loc[(n,arm),'max_drawdown_pct'])]) for n in [1,5,10,20] for arm,name in [('BL_pool','BL'),('Markowitz','Markowitz')]),
    'Cada coluna é uma mediana marginal; não representa necessariamente o mesmo subconjunto.')
same+=f'Para BL, os percentis de retorno em n = 5 são {f(q.loc[5,.1])}\\% e {f(q.loc[5,.9])}\\%; em n = 15, são {f(q.loc[15,.1])}\\% e {f(q.loc[15,.9])}\\%. A amplitude entre esses valores descreve a dependência da composição sorteada. Sua comparação permite avaliar robustez dentro do universo, sem assegurar generalização temporal.\n\n'
same+=r'''Em n = 1, os métodos coincidem por investirem integralmente na mesma ação. Em n = 2, o teto de 50\% impõe pesos-alvo iguais. Para n = 3, o teto de 34,17\% deixa pouca liberdade de alocação. Portanto, as diferenças nos primeiros pontos também refletem restrições geométricas, e não apenas qualidade das estimativas.

'''

selection=r'''\section{Comparação com seleção própria dos ativos}
Na escolha livre, cada modelo ranqueia os ativos pelo retorno esperado excedente dividido pela volatilidade histórica. Markowitz utiliza a média móvel; BL utiliza o posterior. A regra é explicitamente acrescentada à alocação: não corresponde à solução exaustiva do problema combinatório de selecionar a melhor carteira com n ações. Os retornos abaixo incluem seleção e distribuição de pesos.

'''
selection+=table('Desempenho das estratégias com seleção própria.','tab:selecao_propria','rrrrrrr',
    r'n & Ret. BL (\%) & Ret. MV (\%) & Sharpe BL & Sharpe MV & MDD BL (\%) & MDD MV (\%)',
    ''.join(row([n]+[f(gfree[n].loc[a,c]) for c in ['return_pct','sharpe','max_drawdown_pct'] for a in ['BL_pool','Markowitz']]) for n in range(1,21)),
    'Selic no período: 14,87\\%. Há uma trajetória dinâmica por modelo e cardinalidade.')
selection+=f'BL supera Markowitz em retorno para {wins} das 20 cardinalidades. Com uma ação, seu retorno é {f(gfree[1].loc["BL_pool","return_pct"])}\\%, contra {f(gfree[1].loc["Markowitz","return_pct"])}\\% de Markowitz, e seu drawdown é {f(gfree[1].loc["BL_pool","max_drawdown_pct"])}\\%. Nesse caso, a distribuição de pesos é trivial: toda diferença decorre das ações escolhidas e das operações. A concentração exige atenção própria, mesmo quando carteiras maiores apresentam resultados favoráveis.\n\n'
selection+=r'''\subsection*{Trajetórias de capital e comparação com a renda fixa}
\begin{figure}[H]
\centering
\caption{Capital realizado em cardinalidades ilustrativas.}
\includegraphics[width=\textwidth]{05_Resultados/figures/trajetoria de capital.png}
\par\smallskip{\footnotesize Fonte: elaboração própria. Capital inicial de R\$ 100 mil; custos de 10 bps. Os tamanhos são exemplos, não uma seleção de cardinalidade feita antes do teste.}
\end{figure}
'''
selic=float(z.loc['BL_pool','selic_return_pct'])
selection+=table('Resultado com todo o universo de vinte ativos.','tab:universo_completo','lrrr',
    'Indicador & BL & Markowitz & Selic',
    row([r'Capital final (R\$)',f(z.loc['BL_pool','final_capital_brl']),f(z.loc['Markowitz','final_capital_brl']),f(100000*(1+selic/100))])+
    row([r'Retorno líquido (\%)',f(z.loc['BL_pool','return_pct']),f(z.loc['Markowitz','return_pct']),f(selic)])+
    row([r'Volatilidade anual (\%)',f(z.loc['BL_pool','volatility_pct']),f(z.loc['Markowitz','volatility_pct']),'0,00'])+
    row(['Sharpe sobre a Selic',f(z.loc['BL_pool','sharpe']),f(z.loc['Markowitz','sharpe']),'Não definido'])+
    row([r'Drawdown máximo (\%)',f(z.loc['BL_pool','max_drawdown_pct']),f(z.loc['Markowitz','max_drawdown_pct']),'0,00']),
    'A Selic é uma série determinística; seu Sharpe não é definido por ter excesso e volatilidade nulos.')
selection+=f'Em n = 20, os cenários coincidem, pois todos os ativos são elegíveis. A diferença de retorno BL menos Markowitz é {f(z.loc["BL_pool","return_pct"]-z.loc["Markowitz","return_pct"])} p.p.; frente à Selic, é {f(z.loc["BL_pool","return_pct"]-selic)} p.p. As volatilidades são {f(z.loc["BL_pool","volatility_pct"])}\\% e {f(z.loc["Markowitz","volatility_pct"])}\\%, respectivamente. Essa comparação deve considerar também os drawdowns, pois a renda fixa não apresenta as oscilações das ações neste modelo simplificado.\n\n'
selection+=f'O maior retorno de BL na escolha livre ocorre em n = {int(maxret.n)}, com {f(maxret.return_pct)}\\%; seu maior Sharpe ocorre em n = {int(maxsh.n)}, com {f(maxsh.sharpe)}. Esses máximos foram identificados retrospectivamente, não validam um tamanho ótimo antes do teste.\n\n'

costsection=r'\section{Sensibilidade à cardinalidade e aos custos de transação}'+'\n'
costsection+=f'Na escolha livre de BL, os retornos em n = 1, n = 2 e n = 20 são, respectivamente, {f(gfree[1].loc["BL_pool","return_pct"])}\\%, {f(gfree[2].loc["BL_pool","return_pct"])}\\% e {f(z.loc["BL_pool","return_pct"])}\\%. A volatilidade passa de {f(gfree[1].loc["BL_pool","volatility_pct"])}\\% para {f(z.loc["BL_pool","volatility_pct"])}\\% entre os extremos. Não se presume melhora monotônica: aumentar n modifica as ações selecionadas, o risco de concentração e a liberdade de otimização.\n\n'
costsection+=table('Efeito dos custos nas estratégias com seleção própria.','tab:custos','rlrrrr',
    r'n & Modelo & Sem custo (\%) & Líquido (\%) & Impacto (p.p.) & Custo pago (R\$)',
    ''.join(row([n,name]+[f(cost.loc[(n,arm),c]) for c in ['gross','net','impact','cost']]) for n in [1,2,5,10,20] for arm,name in [('BL_pool','BL'),('Markowitz','Markowitz')]),
    'O contrafactual sem custo mantém os mesmos pesos e operações, removendo apenas o desconto financeiro; não há nova otimização.')
costsection+=f'O custo nominal pago e a perda de capital final não são a mesma medida: uma despesa antecipada também reduz o capital sujeito a retornos posteriores. Para n = 20, BL paga R\\$ {f(cost.loc[(20,"BL_pool"),"cost"])}, com impacto de {f(cost.loc[(20,"BL_pool"),"impact"])} p.p. de retorno. Markowitz paga R\\$ {f(cost.loc[(20,"Markowitz"),"cost"])}, com impacto de {f(cost.loc[(20,"Markowitz"),"impact"])} p.p.; seu turnover é {f(z.loc["Markowitz","turnover_total"])}, contra {f(z.loc["BL_pool","turnover_total"])} de BL.\n\n'
costsection+=f'Em n = 2, BL tem turnover de {f(gfree[2].loc["BL_pool","turnover_total"])} e custo de R\\$ {f(cost.loc[(2,"BL_pool"),"cost"])}, com impacto de {f(cost.loc[(2,"BL_pool"),"impact"])} p.p. Embora os pesos-alvo sejam 50/50, a troca das ações escolhidas e a restauração das proporções geram negociação. A comparação entre retorno bruto e líquido permite distinguir desempenho financeiro e despesas de implementação.\n\n'
costsection+='A frequência de rebalanceamento é mensal. A dispersão entre subconjuntos é uma análise de composição, mas não equivale a uma exclusão controlada de um ativo específico. A avaliação de outras frequências e da remoção de ativos exige execução homogênea antes de receber conclusões numéricas.\n\n'

coverage=texts[TCC/rel].split('\\section{Cobertura de notícias')[1].split('\\section{Simulação com o novo')[0]
coverage='\\section{Cobertura de notícias'+coverage
coverage=coverage[:coverage.index('A comparação complementar')]
coverage+=r'A Seção~\ref{sec:etapas_alocacao} apresenta controles com prior sem views e previsão quantitativa diretamente no otimizador. Entretanto, esses controles não equivalem a comparar BL apenas quantitativo, BL apenas textual e híbrido completo sob um protocolo independente. Essa ablação continua necessária para atribuir ganhos a cada componente, especialmente ao sentimento.'+'\n\n'
controls=r'''\section{Avaliação das etapas de alocação}
\label{sec:etapas_alocacao}
A avaliação intermediária distingue a previsão quantitativa, a expectativa posterior do BL e a carteira realizada. Na referência XGBoost direto, a previsão do modelo conjunto é dividida pelo horizonte e descontada da taxa livre de risco, entrando no mesmo otimizador de média--variância sem atualização BL. No prior sem views, utiliza-se a expectativa de equilíbrio, mantendo o otimizador, as restrições, os custos e o calendário. Essa referência não equivale a uma carteira equiponderada de compra e manutenção.

'''
controls+=table('Referências de alocação com vinte ativos.','tab:etapas_alocacao','lrrrr',
    r'Configuração & Retorno (\%) & Sharpe & MDD (\%) & Custo (R\$)',
    ''.join(row([name]+[f(z.loc[arm,c]) for c in ['return_pct','sharpe','max_drawdown_pct','cost_brl']]) for arm,name in [('BL_pool','BL híbrido'),('XGB_pool','XGBoost direto'),('Prior','Prior sem views'),('Markowitz','Markowitz')]),
    'Mesmo universo e condições de execução; Sharpe sobre a referência Selic, que acumula 14,87\\% no período.')
controls+=r'''Com vinte ativos, o XGBoost direto retorna 3,27\%, abaixo da Selic, com Sharpe de -0,58 e drawdown máximo de -20,13\%. O BL apresenta 27,79\%, Sharpe de 0,70 e drawdown de -8,46\%. A diferença mostra que a expectativa quantitativa isolada e a atualização pelo BL produzem alocações distintas; não demonstra, por si só, qualidade elevada do previsor ou superioridade em outro período.

O prior sem views retorna 28,21\%, com Sharpe de 0,74 e custo de R\$ 135,07. Nessa cardinalidade, sua rentabilidade supera a do híbrido em 0,42 p.p. e seu custo é menor. A combinação das views não assegura, portanto, contribuição financeira incremental positiva nesse recorte. A comparação oferece um controle de atribuição, mas não isola todas as interações entre as fontes de informação. O gate de três notícias habilita apenas seis dos 240 pares mensais, e a covariância pode transmitir a influência de uma opinião a outros ativos.

'''
limitations=r'''\section{Limitações e alcance das conclusões}
A primeira limitação é temporal. O treino purga os alvos por maturidade e utiliza 2024 na seleção dos grupos de atributos, mas a escolha de destacar o modelo conjunto ocorreu após o exame de 2025. Os horizontes por ativo seguem o protocolo direcional previamente fixado, sem nova seleção pelo pool. Além disso, o gate de três foi escolhido após observar o desempenho de 2025. O respeito à disponibilidade da informação no motor não elimina esses problemas de seleção. A avaliação é exploratória, não uma comprovação definitiva fora da amostra.

A segunda limitação é estatística. Há apenas um ano de avaliação financeira e vários tamanhos de carteira examinados. Os subconjuntos compartilham ações, datas e choques de mercado, de modo que as 1.142 trajetórias principais e as 1.142 trajetórias de controle não representam amostras independentes. As frequências de vitória são descritivas. A escolha do melhor n pelo retorno de 2025 favoreceria uma configuração ex post; sua utilidade operacional exige validação anterior e teste em outro período, além de procedimentos que respeitem a dependência temporal.

A terceira limitação é de atribuição. Com gate de três notícias, 6 dos 240 pares ativo-data habilitam as views. As referências sem views e com XGBoost direto não separam integralmente os efeitos do prior, das restrições, da covariância e de cada fonte de inteligência computacional. O diagnóstico quantitativo, com R$^2$ negativo em 19 ativos e acurácia superior à classe majoritária em apenas um, reforça a necessidade de evitar inferências causais apenas pelo retorno agregado.

A quarta limitação é operacional. O universo é fixo, está sujeito a viés de sobrevivência e inclui classes de ações da mesma empresa, como PETR3 e PETR4; vinte papéis não equivalem a vinte fontes independentes de risco. A execução ao fechamento seguinte é uma aproximação, os custos são lineares e não se simulam spread, impacto de mercado, tributação ou liquidação terminal. O piso de 0,1\% mantém exatamente n posições, mas algumas podem ser economicamente pequenas.

Por fim, o prior equiponderado evita usar capitalização observada posteriormente, mas se afasta da especificação de equilíbrio ponderado por valor de mercado. A escala diária das views também é aproximada. O referencial de 15\% ao ano é uma taxa constante hipotética, e não a trajetória histórica da Selic. Essas convenções delimitam a comparação e precisam permanecer explícitas no procedimento experimental.

No universo completo, BL retorna 27,79\%, frente aos 19,10\% de Markowitz e aos 14,87\% da Selic, enquanto o prior sem views retorna 28,21\%. O resultado da carteira deve ser interpretado em conjunto com a baixa precisão quantitativa e os controles de atribuição.
'''
texts[TCC/rel]=prefix+quant+same+selection+costsection+coverage+controls+limitations

# Metodologia: a configuracao conjunta e a unica utilizada no experimento.
proced='04_Procedimentos/procedimento.tex'
new_method=r'''\subsection{Estimador quantitativo XGBoost}

A etapa quantitativa utiliza regressão \textit{eXtreme Gradient Boosting} (XGBoost), com um modelo conjunto das vinte ações para cada horizonte de cinco, dez e vinte e um pregões. A saída é o log-retorno prospectivo estimado. A identidade do ativo é incluída nas entradas por codificação indicadora, permitindo aprender relações comuns e diferenças entre as ações. O sinal da previsão numérica é utilizado no cálculo da acurácia direcional, sem um classificador separado.

O treinamento emprega janela expansiva, duas divisões semestrais de validação em 2024 e quatro trimestrais de avaliação em 2025. Em cada divisão, entram somente alvos conhecidos antes do início do período previsto. Há uma divisão interna de 80\% para ajuste e 20\% para interrupção antecipada, também purgada pela maturidade dos alvos. Os filtros de atributos utilizam apenas o ajuste interno, sem consultar os desfechos da avaliação. A semente de aleatoriedade é fixada em 42.

A função de perda é quadrática, com profundidade máxima quatro, taxa de aprendizado 0,03, subamostragem de linhas e colunas de 0,8, peso mínimo de folha de 10, regularizações L1 de 0,1 e L2 de 1 e pesos uniformes entre observações. O limite é de quatrocentas árvores, com interrupção após vinte rodadas sem melhora do RMSE. O número de árvores é definido nessa etapa, seguido de novo ajuste com todos os alvos já conhecidos no início da divisão. O alvo é padronizado pelas estatísticas do treinamento e a previsão é reconstruída em log-retorno antes da avaliação.

Os grupos de atributos por horizonte são selecionados em 2024 pelo desempenho relativo à referência de retorno zero: conjunto reduzido em cinco pregões, reduzido com novas variáveis em dez e ampliado com medidas transversais de mercado em vinte e um. Os horizontes utilizados por ativo seguem uma seleção direcional em 2024 previamente fixada no protocolo, e não uma nova escolha pelo desempenho do pool em 2025. O desempenho do pool na validação é reportado separadamente desse critério. A escolha da arquitetura conjunta ocorreu após o exame de 2025; por isso, a avaliação permanece exploratória e exige confirmação independente.

Na avaliação preditiva, são consideradas somente previsões com desfecho conhecido antes de 31 de dezembro de 2025. Na simulação financeira, incluem-se também sinais cujo desfecho ocorre depois do fim do ano; o resultado futuro não entra no treino nem na confiança antes de sua maturidade. A confiança quantitativa é calculada pelos erros absolutos já maturados, normalizados por sua mediana histórica, sem penalização direcional adicional.

'''
block(proced,'\\subsection{Estimador quantitativo XGBoost}','\\subsection{Modelo de linguagem',new_method)
once(proced,'A rodada consolidada e os testes complementares são comparados pelas métricas isoladas, sem utilizar esse escore para selecionar retrospectivamente a melhor configuração.',
    'As estratégias são comparadas pelas métricas isoladas, sem utilizar esse escore para selecionar retrospectivamente a melhor configuração.')
once(proced,'no critério original de eleição dos modelos. Na rodada complementar de aperfeiçoamento, a seleção de atributos e métodos utiliza o erro quantitativo em 2024, mantendo o acerto direcional como diagnóstico.',
    'na regra de horizontes previamente fixada. A seleção dos grupos de atributos utiliza erro quantitativo em 2024, mantendo o acerto direcional como diagnóstico.')
once(proced,'A arquitetura permite frequências diária, semanal e mensal, mas a rodada consolidada e a comparação complementar utilizam somente a frequência mensal. Os sinais são formados no primeiro pregão de cada mês e executados no fechamento seguinte. Resultados de outras frequências obtidos por motores anteriores não são tratados como sensibilidade diretamente comparável.',
    'A avaliação utiliza a frequência mensal. Os sinais são formados no primeiro pregão de cada mês e executados no fechamento seguinte. Outras frequências não integram a evidência financeira apresentada.')
once(proced,'A seleção dos modelos e dos parâmetros ocorre em dois níveis. No componente quantitativo, a busca Optuna explora configurações do XGBoost sobre divisões temporais de treinamento, enquanto a escolha final de horizonte e tipo de estimador utiliza o desempenho direcional OOF. No componente de alocação, o Optuna avalia parâmetros do Black--Litterman, incluindo $\\alpha$, $\\tau$, teto de concentração, pesos relativos das fontes de confiança e intervalos de limitação da incerteza. A função objetivo adotada na busca do portfólio é a estatística de Sharpe realizada no período de validação.',
    'A seleção dos parâmetros ocorre em dois níveis. No componente quantitativo, os grupos de atributos são escolhidos em 2024, mantendo a configuração de regressão conjunta descrita anteriormente e os horizontes já fixados no protocolo. No componente de alocação, os valores de $\\alpha$, $\\tau$, teto de concentração e pesos das fontes de confiança são mantidos fixos durante a simulação. A escolha do modelo conjunto e do gate utilizou diagnósticos de 2025, não constituindo seleção integralmente anterior à avaliação.')
once(proced,'A seleção original dos artefatos quantitativos pode ter consultado o OOF completo; por isso, a separação de anos não torna a avaliação integralmente independente.',
    'A arquitetura conjunta foi escolhida após o exame de 2025; por isso, a separação de anos não torna a avaliação integralmente independente.')
once(proced,'As comparações consolidadas variam a cardinalidade e os subconjuntos de ativos. A rodada complementar substitui o XGBoost, mantendo as condições financeiras, e inclui controles com confiança original e sem atualização BL. A variação de frequência e a exclusão controlada de ativos permanecem extensões que exigem execução homogênea, sem autorizar a escolha retrospectiva da melhor configuração em 2025.',
    'As comparações variam a cardinalidade e os subconjuntos de ativos. A análise principal de BL e Markowitz reúne 1.142 trajetórias; as referências de prior sem views e XGBoost conjunto diretamente no otimizador acrescentam 1.142 trajetórias. São 2.284 trajetórias apresentadas, sem independência estatística entre elas. Na referência direta, utiliza-se a previsão dividida pelo horizonte e descontada a taxa livre de risco, mantendo custos, calendário, restrições e estimação da covariância. A variação de frequência e a exclusão controlada de ativos permanecem extensões que exigem execução homogênea.')
once(proced,'A arquitetura descreve pesos de equilíbrio por capitalização, mas a rodada consolidada utiliza prior equiponderado para evitar capitalizações pontuais posteriores;',
    'O experimento utiliza prior equiponderado para evitar capitalizações pontuais posteriores;')
once(proced,'A escolha posterior do gate e o destaque do pool','A escolha do gate e da arquitetura conjunta após observar 2025')

# Conclusao conserva a organizacao e os paragrafos que nao dependem da troca.
con='06_Conclusao/conclusao.tex'
block(con,'No componente quantitativo,','No componente qualitativo,',
    'No componente quantitativo, o trabalho considerou regressão conjunta das vinte ações e previsões em diferentes horizontes. A escolha entre cinco, dez e vinte e um pregões permitiu representar escalas temporais distintas, em vez de impor um único horizonte a todo o universo. Os horizontes por ativo seguem a seleção direcional em 2024 previamente fixada no protocolo. A utilização de previsões produzidas pelo procedimento \\emph{walk-forward}, com purga pela maturidade dos alvos, possibilitou avaliar o comportamento do estimador em segmentos posteriores às suas janelas de treinamento. A escolha da arquitetura após observar 2025 limita o caráter confirmatório dessa avaliação.\n\n')
block(con,'A avaliação do portfólio ampliou','Dessa forma,',
    'A avaliação do portfólio examinou cardinalidades de um a vinte ativos em dois cenários: subconjuntos comuns aos modelos e seleção própria por estratégia. Essa separação permitiu investigar o desempenho da alocação sob oportunidades equivalentes e, complementarmente, o resultado do processo completo de seleção e distribuição de pesos. Foram apresentadas 1.142 trajetórias de BL e Markowitz e 1.142 trajetórias de controle com prior sem views e previsão quantitativa diretamente no otimizador, com custos de transação e defasagem de execução.\n\n')
once(con,'retorno líquido acumulado de 28,21\\%, contra','retorno líquido acumulado de 27,79\\%, contra')
once(con,'aproximadamente 0,73 para o modelo híbrido','aproximadamente 0,70 para o modelo híbrido')
block(con,'Os resultados quantitativos recomendam','A cobertura efetivamente utilizada',
    f'''Os resultados quantitativos recomendam cautela na interpretação dos ganhos da carteira. A acurácia direcional média entre ativos, nos horizontes utilizados pela carteira, foi de {f(hr)}\\% em 2025, e dezenove dos vinte ativos apresentaram coeficiente de determinação negativo. No horizonte de 21 pregões, considerando as vinte ações, o MAE médio foi de 6,72 e o RMSE de 8,42, em log-retorno multiplicado por 100. A acurácia direcional foi de 58,41\\%, mas o R$^2$ médio por ativo foi de -0,1354. Portanto, o desempenho financeiro observado não corresponde a uma demonstração de elevada precisão na previsão das magnitudes dos retornos.

A avaliação das etapas reforça essa distinção. Com vinte ativos, a previsão quantitativa diretamente no otimizador apresentou retorno de 3,27\\%, Sharpe de -0,58 e drawdown máximo de -20,13\\%, permanecendo abaixo da Selic. O BL apresentou retorno de 27,79\\%, Sharpe de 0,70 e drawdown de -8,46\\%. A carteira depende conjuntamente do prior, da covariância, das restrições, das views e de sua confiança, o que impede atribuir seu resultado a uma única fonte de informação.

''')
once(con,'próximo ao BL original e com menor custo de negociação.',
    'superior aos 27,79\\% do híbrido e com menor custo de negociação.')
once(con,'os controles acrescentam evidência sobre o previsor e o prior,','os controles acrescentam evidência sobre as etapas de alocação e o prior,')
once(con,'Somam-se a isso a seleção original dos estimadores quantitativos, que ainda exige validação temporal integral, o destaque do pool após observar 2025, o universo fixo de ações e as simplificações de execução e custos. O prior equiponderado da rodada também deve ser distinguido da especificação ponderada por capitalização apresentada na arquitetura.',
    'Somam-se a isso a escolha da arquitetura conjunta após observar 2025, a manutenção dos horizontes previamente definidos, o universo fixo de ações e as simplificações de execução e custos. O prior equiponderado também delimita a interpretação do equilíbrio utilizado.')
once(con,'Isso inclui a escolha entre classificador e regressor, o horizonte por ativo,','Isso inclui a configuração da regressão conjunta, o horizonte por ativo,')
once(con,'A receita do pool deve permanecer fixada antes dessa avaliação, mantendo a comparação com o modelo original e com referências simples de previsão.',
    'A configuração do modelo conjunto deve permanecer fixada antes dessa avaliação, mantendo referências simples de previsão, como retorno zero e média histórica disponível na origem.')
once(con,'Os controles iniciais com prior sem views e substituição do XGBoost',
    'Os controles com prior sem views e XGBoost diretamente no otimizador')

# Resumo e abstract apresentam somente o experimento final.
once('01_Pre_textuais/Resumo.tex','Os testes complementares reduziram os erros do XGBoost, mas o coeficiente de determinação médio permaneceu negativo e o efeito financeiro foi misto.',
    'O XGBoost conjunto apresentou coeficiente de determinação médio negativo, enquanto o BL alcançou retorno líquido de 27,79\\% com vinte ativos, superior a Markowitz e à Selic, mas inferior ao prior sem views.')
once('01_Pre_textuais/Abstract.tex','Additional tests reduced XGBoost prediction errors, but the average coefficient of determination remained negative and the financial effect was mixed.',
    'The pooled XGBoost model yielded a negative average coefficient of determination, while BL achieved a net return of 27.79\\% with twenty assets, exceeding Markowitz and Selic but falling below the prior without views.')

# Fundamento: direcao e extraida da regressao; nao afirmar uso de classificador.
once('02_Fundamentacao/fundamentacao_teorica.tex',
    'Neste trabalho, essas duas formas de aprendizado são utilizadas de maneira complementar. A regressão é empregada para estimar a magnitude dos retornos futuros dos ativos, enquanto a classificação é utilizada para determinar sua direção esperada (movimento de alta ou baixa).',
    'Neste trabalho, a regressão é empregada para estimar a magnitude dos retornos futuros dos ativos. A direção esperada (movimento de alta ou baixa) é obtida pelo sinal da previsão numérica e utilizada como diagnóstico, sem exigir um classificador separado.')

arch='03_Arquitetura/arquitetura.tex'
once(arch,'A arquitetura descrita neste capítulo apresenta o funcionamento geral do sistema e sua versão original de treinamento individual. Na rodada consolidada, o procedimento experimental utiliza prior equiponderado e decisões mensais. Os testes complementares acrescentam o treinamento conjunto por horizonte, mantendo a saída quantitativa e a integração com o BL. Essas adaptações e seu caráter exploratório são delimitados no capítulo de procedimento e avaliados no capítulo de resultados.',
    'A arquitetura descrita neste capítulo utiliza regressão XGBoost conjunta das vinte ações por horizonte, mantendo uma saída quantitativa por ativo e sua integração com o BL. O experimento adota prior equiponderado e decisões mensais. As condições de treinamento, os critérios de seleção e o caráter exploratório da avaliação são delimitados no capítulo de procedimento e avaliados no capítulo de resultados.')
once(arch,'Para cada ativo e horizonte, são treinadas instâncias independentes dos modelos preditivos. A escolha do horizonte vencedor é realizada com base no desempenho direcional obtido fora da amostra, utilizando previsões \\textit{out-of-fold}.',
    'Para cada horizonte, é treinado um regressor conjunto das vinte ações, com identidade do ativo entre as entradas. Os horizontes utilizados por ativo seguem a seleção direcional em 2024 já fixada no protocolo, sem nova eleição pelo desempenho do pool em 2025.')
once(arch,'A seleção do horizonte é realizada exclusivamente com base nas observações disponíveis na etapa de treinamento de cada janela temporal. O conjunto de teste permanece separado até a avaliação final, evitando que a escolha do horizonte introduza viés otimista nos resultados.',
    'A definição dos horizontes utiliza observações anteriores a 2025 e alvos já maturados. Entretanto, a arquitetura conjunta foi escolhida após o exame de 2025; o respeito à ordem temporal no treinamento não torna a seleção integralmente independente da avaliação.')
once(arch,'Quando a acurácia direcional do modelo selecionado é inferior a $52\\%$, sua confiança é reduzida por meio de um fator de penalização incorporado ao cálculo da matriz $\\boldsymbol{\\Omega}$. Esse procedimento não elimina necessariamente o sinal quantitativo, mas reduz sua influência na atualização posterior do modelo Black--Litterman.',
    'A confiança quantitativa utiliza somente erros de previsões com horizonte já encerrado. A acurácia direcional é apresentada como diagnóstico, sem uma penalização adicional de confiança baseada em seu limiar.')
start='Durante o treinamento, as observações mais recentes recebem maior peso'
end='O resultado desse processo é uma previsão quantitativa de retorno'
block(arch,start,end,
    'Durante o treinamento, as observações recebem pesos uniformes. Os modelos utilizam perda quadrática, regularização e interrupção antecipada pelo RMSE. Os grupos de atributos são selecionados em 2024, com filtros calculados apenas no treinamento.\n\nO componente quantitativo utiliza um regressor conjunto por horizonte. A previsão numérica é reconstruída em log-retorno após a padronização interna do alvo, e sua direção é dada pelo sinal desse valor. Não há conversão de probabilidades de classes em retorno.\n\n')
block(arch,'em que $e_{i,t}$ representa o erro preditivo recente,','A confiança associada às notícias é calculada',
    'em que $e_{i,t}$ representa o erro absoluto médio das previsões já maturadas, $\\xi_{i,t}$ representa a mediana dos erros absolutos desse histórico e $\\kappa_{i,t}^{\\mathrm{acc}}=1$. A escala é limitada inferiormente para estabilidade numérica. A confiança não utiliza o resultado futuro da previsão em curso nem aplica penalização adicional por acurácia direcional.\n\n')
block(arch,'O \\textit{prior} de mercado representa o vetor de retornos de equilíbrio implícitos nos pesos de mercado.','O vetor de retornos de equilíbrio é então calculado por:',
    'O \\textit{prior} representa o vetor de retornos de equilíbrio utilizado como referência. Para evitar o uso de capitalizações observadas posteriormente, adota-se um vetor equiponderado $\\boldsymbol{w}_{t}^{\\mathrm{mkt}}=\\boldsymbol{1}/N$. A covariância utiliza janela móvel de 126 pregões, disponível na data da decisão, e regularização diagonal de $10^{-7}$. O coeficiente de aversão ao risco é fixado em $\\delta_t=3$.\n\n')
once(arch,'Essa formulação ancora as expectativas do modelo nos retornos implícitos pela estrutura de mercado. Dessa maneira, os sinais algorítmicos não substituem completamente as informações agregadas nos preços e nos pesos de capitalização, mas são utilizados para ajustar o equilíbrio de acordo com o nível de confiança atribuído a cada opinião.',
    'Essa formulação ancora as expectativas em um equilíbrio equiponderado, condicionado à covariância histórica e à aversão ao risco. Os sinais algorítmicos ajustam essa referência de acordo com o nível de confiança atribuído a cada opinião.')

# Grava somente apos todas as substituicoes esperadas terem sido verificadas.
changed=[];diffs=[]
for p,new in texts.items():
    old=original[p].decode('utf-8-sig').replace('\r\n','\n')
    if old==new:continue
    changed.append(p)
    diffs.extend(difflib.unified_diff(old.splitlines(keepends=True),new.splitlines(keepends=True),
        fromfile=p.relative_to(TCC).as_posix()+' (antes)',tofile=p.relative_to(TCC).as_posix()+' (depois)'))
    p.write_bytes(new.replace('\n','\r\n' if b'\r\n' in original[p] else '\n').encode('utf-8'))
(OUT/'alteracoes.diff').write_text(''.join(diffs),encoding='utf-8')
m.to_csv(OUT/'metricas_utilizadas.csv',index=False)
scores.to_csv(OUT/'previsoes_horizontes_utilizados.csv',index=False)
cost.reset_index().to_csv(OUT/'custos_utilizados.csv',index=False)
paired.to_csv(OUT/'comparacao_pareada.csv')
(OUT/'arquivos_alterados.json').write_text(json.dumps([str(p) for p in changed],ensure_ascii=False,indent=2),encoding='utf-8')

os.environ.setdefault('MPLCONFIGDIR',str(REPO/'tmp/xgb_matplotlib_cache'))
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
plt.rcParams.update({'font.family':'DejaVu Sans','font.size':10,'axes.spines.top':False,'axes.spines.right':False})
colors={'BL_pool':'#2468a0','Markowitz':'#c7742e'}
fig,ax=plt.subplots(figsize=(9,4.8))
for arm,name in [('BL_pool','Black-Litterman'),('Markowitz','Markowitz')]:
    g=common[common.model.eq(arm)].groupby('n').return_pct
    s=g.agg('median');lo=g.quantile(.1);hi=g.quantile(.9)
    ax.plot(s.index,s.values,color=colors[arm],label=name,lw=2)
    ax.fill_between(s.index.to_numpy(),lo.to_numpy(),hi.to_numpy(),color=colors[arm],alpha=.16)
ax.axhline(selic,color='#555',ls=':',label='Selic hipotética')
ax.set(xlabel='Número de ativos',ylabel='Retorno líquido (%)',xticks=[1,5,10,15,20])
ax.legend();ax.grid(alpha=.15);fig.tight_layout()
fig.savefig(TCC/'05_Resultados/figures/retorno liquido ativos.png',dpi=200);plt.close(fig)
fig,axes=plt.subplots(1,3,figsize=(13,4.8))
for ax,n in zip(axes,[5,10,20]):
    for arm,name in [('BL_pool','Black-Litterman'),('Markowitz','Markowitz')]:
        c=curve[curve.n.eq(n)&curve.model.eq(arm)].sort_values('date')
        ax.plot(c.date,c.capital/1000,color=colors[arm],label=name,lw=1.8)
    dates=c.date
    ax.plot(dates,100*(1.15)**(np.arange(1,len(dates)+1)/252),ls=':',color='#555',label='Selic hipotética')
    ax.set_title(f'n = {n}');ax.set_ylabel('Capital (R$ mil)');ax.tick_params(axis='x',rotation=30);ax.grid(alpha=.15)
axes[0].legend(fontsize=8);fig.tight_layout()
for name in ['trajetoria de capital.png','pool_trajetorias.png']:
    fig.savefig(TCC/'05_Resultados/figures'/name,dpi=200)
plt.close(fig)
print('Pasta de auditoria:',OUT)
print('Acuracia nos horizontes utilizados:',hr)
print('BL 20 ativos:',z.loc['BL_pool'].to_dict())
print('Arquivos de texto atualizados:',len(changed))
