"""Integra as camadas no capitulo fornecido pelo usuario, preservando as tabelas financeiras."""
from pathlib import Path
import hashlib
import json
import shutil
import pandas as pd
from .camadas_intermediarias import OUT, ROOT
from .avaliar_predicao_xgb import fmt

SOURCE=Path(r'C:\Users\igor.toledo_visagio\.codex\attachments\ba1131df-9214-43fb-856f-4dad0d3fe2ce\Texto colado.txt')
TCC=Path(r'C:\Users\igor.toledo_visagio\Documents\TCC 2')
DEST=OUT/'capitulo_integrado'
CODES=dict(XGB='XGB',Q_neutro=r'$Q_0$',Q=r'$Q$',BL_XGB=r'$BL_X$',BL_neutro=r'$BL_0$',BL='BL',Prior=r'$\pi$')


def table(caption,label,headers,rows,spec,note='Fonte: elaboração própria.'):
    return '\n'.join([r'\begin{table}[H]\centering',r'\caption{'+caption+'}',r'\label{'+label+'}',
        r'{\footnotesize\setlength{\tabcolsep}{4pt}\begin{tabular}{'+spec+'}\\hline',
        ' & '.join(headers)+r'\\ \hline',*[' & '.join(map(str,r))+r'\\' for r in rows],
        r'\hline\end{tabular}}',r'\par\smallskip{\footnotesize '+note+'}',r'\end{table}'])


def figure(name,caption,label):
    return '\n'.join([r'\begin{figure}[H]\centering',r'\caption{'+caption+'}',r'\label{'+label+'}',
        r'\includegraphics[width=0.95\textwidth]{05_Resultados/predicao_xgb/'+name+'}',
        r'\par\smallskip{\footnotesize Fonte: elaboração própria.}\end{figure}'])


def main():
    DEST.mkdir(parents=True,exist_ok=True)
    chapter=TCC/'05_Resultados/resultados.tex';fragment=TCC/'05_Resultados/predicao_xgb.tex'
    # Guardar as versoes anteriores sem sobrescrever backups em reexecucoes.
    for path in (chapter,fragment):
        digest=hashlib.sha256(path.read_bytes()).hexdigest()[:12]
        backup=DEST/(path.stem+'_anterior_'+digest+'.tex')
        if not backup.exists():shutil.copyfile(path,backup)
    base=SOURCE.read_text(encoding='utf-8-sig')
    xgb=pd.read_csv(OUT/'todas_previsoes/xgb_resumo_horizontes.csv')
    quarterly=pd.read_csv(OUT/'predicao_xgb/trimestres_selecionados.csv')
    scores=pd.read_csv(OUT/'qualidade_camadas/scores.csv')
    main_scores=scores[(scores.target=='actual_horizon_daily')&(scores.scope=='todos')]
    next_day=scores[(scores.target=='actual_next_day')&(scores.scope=='todos')]
    effects=pd.read_csv(OUT/'qualidade_camadas/efeito_pareado.csv')
    effects=effects[(effects.target=='actual_horizon_daily')&(effects.scope=='todos')]
    fin=pd.read_csv(OUT/'qualidade_camadas/financeiro_controlado.csv')
    base_preds=pd.read_csv(OUT/'predicao_xgb/metricas.csv')
    selected=base_preds[base_preds.selected & base_preds.period.eq('2025')]
    wins=int((selected.hr>selected.hr_majority+1e-10).sum())
    # Um unico fragmento aberto: com subsecoes quantitativas e nova secao preditiva.
    text=[r'\subsection{Diagnóstico do histórico completo e comparação com referências}',
        'Para contextualizar a qualidade da inteligência quantitativa, o inventário reúne 41.852 previsões '
        'out-of-fold (OOF) de vinte ativos em três horizontes, entre 14 de fevereiro de 2023 e 31 de março de 2026. '
        'Os vinte arquivos dos vencedores são cópias das séries por horizonte e não são contabilizados novamente. '
        'O histórico completo é um diagnóstico dos artefatos de treinamento; a janela financeira deste capítulo permanece restrita a 2025.',
        table('Desempenho das previsões OOF no histórico completo, por horizonte.','tab:xgb_historico_completo',
              ['h','Previsões',r'HR (\%)',r'MAE$\times100$',r'RMSE$\times100$',r'R$^2$ médio'],
              [[int(r.horizon),f'{int(r.predictions):,}'.replace(',','.'),fmt(r.hr),fmt(100*r.mae),fmt(100*r.rmse),fmt(r.r2,3)] for r in xgb.itertuples()], 'rrrrrr',
              'Fonte: elaboração própria. Métricas em médias simples entre os vinte ativos. Erros multiplicados por 100 em unidades de log-retorno.'),
        'A acurácia média aumenta de 51,74\\% para 54,73\\% entre os horizontes de cinco e 21 pregões, '
        'enquanto o R-quadrado médio passa de -0,033 para -0,112. '
        'O contraste mostra que maior acerto de direção não implica melhor estimativa da magnitude. '
        'Os valores brutos de erro dos diferentes horizontes correspondem a alvos distintos e não permitem, isoladamente, escolher o melhor horizonte.',
        'A reconciliação dos arquivos identificou 25 combinações em que o resumo histórico conserva MAE, RMSE e R-quadrado '
        'do regressor, embora a série exportada pertença ao classificador vencedor. '
        'As métricas da Tabela~\\ref{tab:xgb_historico_completo} foram recalculadas diretamente dos pares de previsão e resultado efetivamente salvos. '
        'Os arquivos originais foram preservados. Essa correção de mensuração não corresponde a retreinamento.',
        f'No recorte de 2025, avalia-se também a direção mais frequente dos retornos históricos já realizados, '
        f'calculada em janela expansiva com no mínimo 80 alvos conhecidos. Essa referência alcança {fmt(selected.hr_majority.mean())}\\% de acurácia média, '
        f'contra {fmt(selected.hr.mean())}\\% do XGBoost; o modelo a supera em {wins} ativos. '
        'O sinal da média histórica alcança 54,95\\%. Diferentemente da maior classe do próprio teste, '
        'essas referências utilizam somente alvos já conhecidos no fechamento da previsão. '
        'O RMSE do XGBoost é inferior ao da previsão de retorno zero em seis ativos e ao da média histórica em nove.',
        table('Estabilidade trimestral das previsões selecionadas para 2025.','tab:xgb_estabilidade',
              ['Trimestre',r'HR XGB (\%)',r'HR maj. histórica (\%)',r'HR média histórica (\%)'],
              [[str(r.period).replace('2025Q','2025 / T'),fmt(r.hr),fmt(r.hr_majority),fmt(r.hr_historical)] for r in quarterly.itertuples()], 'lrrr'),
        'O modelo supera as referências no segundo trimestre, mas perde essa vantagem no segundo semestre. '
        'No quarto trimestre, sua acurácia é 50,95\\%, contra 67,78\\% da direção majoritária histórica. '
        'Os trimestres são definidos pela data de origem da previsão, e os desfechos podem terminar no trimestre seguinte, '
        'desde que conhecidos até o encerramento do teste. As médias atribuem o mesmo peso a cada ativo.',
        r'\section{Qualidade preditiva das etapas intermediárias}',r'\label{sec:etapas_preditivas}',
        'A avaliação financeira descreve o processo completo, mas não identifica por si só qual componente gera previsões melhores. '
        'Por isso, esta seção confronta o resultado realizado com a previsão quantitativa, com a opinião Q e com o posterior do BL. '
        'As comparações respondem a três perguntas: a mistura com notícias melhora a previsão; a atualização BL melhora a previsão quantitativa; '
        'e o sistema completo acrescenta informação frente ao prior sem opiniões.',
        r'\subsection{Configurações e controles utilizados}',
        table('Configurações utilizadas para avaliar as camadas intermediárias.','tab:configuracoes_camadas',
              ['Código','Configuração e finalidade'],[
                  ['XGB','Previsão quantitativa sem atualização BL.'],
                  [r'$Q_0$','Mistura com polaridade textual zerada; preserva o peso quantitativo.'],
                  [r'$Q$','Opinião híbrida completa, antes do BL.'],
                  [r'$BL_X$','Atualização BL recebendo somente a previsão XGB como opinião.'],
                  [r'$BL_0$','Atualização BL recebendo a opinião com polaridade textual zerada.'],
                  ['BL','Atualização BL recebendo a opinião híbrida completa.'],
                  [r'$\pi$','Prior sem incorporação de opiniões.']],r'lp{0.77\linewidth}'),
        'Nos controles com BL, a covariância, o prior e a matriz de incerteza são os mesmos da rodada original. '
        'No controle de polaridade zerada, mantém-se a quantidade de manchetes, o gate, a confiança e o peso quantitativo dependente da cobertura; '
        'retira-se somente a parcela numérica associada ao sentimento. Portanto, esse controle não representa retirar todas as informações de notícias: '
        'a quantidade de registros ainda influencia a confiança. A nomenclatura evita confundir polaridade zerada com a utilização integral da previsão quantitativa.',
        r'Com $c=\min(1,n_{not}/3)$ e $a(c)=1-(1-\alpha)c$, a opinião no horizonte nativo pode ser escrita como '
        r'$Q_h=a(c)\widehat r_h+(1-\alpha)c\rho_h\psi$. '
        r'No controle $Q_0$, fixa-se $\psi=0$, mantendo $a(c)$; em $BL_X$, a opinião de entrada é a previsão inteira $\widehat r_h$. '
        'Assim, as comparações controladas distinguem retirar polaridade, modificar a mistura e aplicar o BL.',
        r'\subsection{Alvos, métricas e score de previsão}',
        'Dos 240 pares ativo-data nas decisões mensais, 228 possuem desfechos de h pregões concluídos até 30 de dezembro de 2025. '
        'As sete configurações são avaliadas nesses mesmos 228 pares; os doze restantes são excluídos de todas elas. '
        'O conjunto contém quarenta pares com alguma notícia e seis com gate ativo. '
        'A utilização das datas mensais corresponde aos sinais efetivamente consultados pelo processo de alocação, '
        'e não a uma nova avaliação financeira com 228 pregões.',
        r'O alvo comparável é $z_{i,t}=\log(P_{i,t+h}/P_{i,t})/h-r_f^{dia}$. '
        'A previsão XGB e Q são convertidas à mesma escala diária de excesso; prior e posterior mantêm a escala diária do motor. '
        'Essa harmonização entre log-retornos e estimativas baseadas em retornos simples é aproximada, '
        'não equivale a treinar um previsor diário e deve ser considerada ao interpretar os níveis de erro. '
        'Um ponto-base (pb) corresponde a 0,0001 na escala de retorno.',
        r'Para o erro $e=\widehat z-z$, calculam-se MAE, RMSE, acurácia direcional (HR), acurácia balanceada (BA) e R-quadrado. '
        'HR mede o acerto do sinal do excesso de retorno, e BA é a média das taxas de acerto nas classes de sinal presentes. '
        'Por isso, esses acertos não devem ser confundidos com os acertos de retorno bruto da avaliação diária do XGBoost. '
        'O RMSE dá maior peso a erros grandes; a acurácia ignora a magnitude.',
        r'Adota-se o score de habilidade $S=1-\mathrm{MSE}_{modelo}/\mathrm{MSE}_{zero}$, '
        'em que a referência prevê excesso de retorno zero, correspondente à taxa livre de risco nessa escala. '
        'S positivo indica menor erro quadrático que essa referência, S nulo indica empate e S negativo indica piora. '
        'O score não é uma probabilidade nem uma nota limitada ao intervalo de zero a cem. '
        r'Apresenta-se $100S$ para facilitar a leitura.',
        'O R-quadrado usa como referência a média retrospectiva dos alvos da amostra, enquanto S usa excesso zero. '
        'São referências diferentes. Uma previsão pode acertar a direção e ainda ter R-quadrado negativo. '
        'A correlação entre duas camadas também não mede o acerto em relação ao resultado realizado; '
        'por isso, não é usada como score principal de qualidade.',
        table('Scores das previsões nas mesmas 228 observações de 2025.','tab:scores_camadas',
              ['Config.',r'HR (\%)',r'BA (\%)','MAE (pb)','RMSE (pb)',r'$100S$',r'R$^2$'],
              [[CODES[r.layer],fmt(r.hr),fmt(r.balanced_accuracy),fmt(r.mae*10000),fmt(r.rmse*10000),fmt(r.skill_zero*100),fmt(r.r2,3)] for r in main_scores.itertuples()], 'lrrrrrr',
              'Fonte: elaboração própria. Alvo: excesso de log-retorno diário equivalente no horizonte selecionado. Menor MAE/RMSE e maior HR/BA/S/R-quadrado são melhores.'),
        'O BL completo apresenta o menor RMSE, 53,82 pb/dia, e a maior HR, 55,70\\%. '
        'Sua acurácia balanceada é 52,37\\%, frente a 50,00\\% do prior, evidenciando que parte da HR bruta decorre da distribuição de sinais. '
        'O score de 2,12 indica redução de aproximadamente 2,12\\% do MSE frente ao excesso zero, '
        'e não um acerto de 2,12\\%. O R-quadrado de 0,018 mostra que a capacidade explicativa permanece pequena, '
        'mesmo na configuração que apresenta o melhor resultado relativo.',
        r'\subsection{Contribuição incremental das notícias e da atualização BL}',
        table('Efeito incremental das camadas sobre o erro e o acerto de sinal.','tab:efeito_incremental',
              ['Comparação',r'Redução do MSE (\%)','Ganho de HR (p.p.)'],
              [[CODES[r.before]+r' $\rightarrow$ '+CODES[r.after],fmt(r.mse_reduction_pct),fmt(r.hr_gain_pp)] for r in effects.itertuples()], 'lrr',
              'Fonte: elaboração própria. Cada contraste utiliza os mesmos 228 pares. Reduções negativas de MSE indicam piora.'),
        r'A passagem de $Q_0$ para Q aumenta o MSE em 2,07\% e eleva a HR em 0,88 p.p. '
        'Assim, a polaridade textual melhora ligeiramente a direção, mas piora a magnitude antes do BL. '
        r'No posterior, a passagem de $BL_0$ para BL reduz o MSE em 1,73\% e eleva a HR em 4,82 p.p. '
        'O efeito das notícias depende, portanto, da forma de integração; não se observa melhora uniforme em todas as etapas.',
        r'Comparar XGB e $BL_X$ isola a atualização BL sob a mesma confiança da rodada: '
        'o MSE cai 9,81\\% e a HR cresce 7,02 p.p. '
        'A transformação de Q em posterior reduz o MSE em 12,28\\%. '
        'Por outro lado, a melhora do BL completo contra o prior é de apenas 1,34\\% do MSE e 1,32 p.p. de HR. '
        'Essa diferença menor indica que uma parcela relevante do desempenho preditivo final já está presente no prior.',
        'Nos quarenta pares com notícias, a polaridade aumenta o MSE de Q em 8,78\\%, mas reduz o MSE do posterior em 2,96\\%. '
        'A influência sobre outros ativos do mesmo rebalanceamento pode ser transmitida pela covariância. '
        'Os seis pares com gate ativo são insuficientes para uma conclusão estatística autônoma sobre esse subconjunto. '
        'Ausência de significância demonstrada não equivale a ausência de efeito; nesta análise, os efeitos são descritos, sem teste confirmatório.',
        figure('scores_camadas.png','Qualidade preditiva das configurações no mesmo alvo e nas mesmas observações.','fig:scores_camadas'),
        r'\subsection{Sensibilidade ao alvo de um pregão}',
        'Como o BL opera na frequência diária, verifica-se também o retorno simples excedente do fechamento do sinal ao fechamento seguinte, '
        'nos 240 pares mensais. As previsões de XGB e Q são convertidas do log-retorno diário equivalente para retorno simples por exponenciação. '
        'Essa é uma pergunta preditiva distinta, pois o XGBoost foi treinado para vários pregões.',
        table('Sensibilidade dos scores ao alvo de retorno excedente no pregão seguinte.','tab:scores_pregao_seguinte',
              ['Config.',r'HR (\%)',r'BA (\%)','RMSE (pb)',r'$100S$'],
              [[CODES[r.layer],fmt(r.hr),fmt(r.balanced_accuracy),fmt(r.rmse*10000),fmt(r.skill_zero*100)] for r in next_day[next_day.layer.isin(['XGB','Q','BL','Prior'])].itertuples()], 'lrrrr',
              'Fonte: elaboração própria. N = 240 pares nas quatro configurações.'),
        'No alvo de um pregão, o BL completo mantém o menor RMSE, mas sua HR de 49,17\\% fica abaixo dos 50,00\\% de XGB e Q. '
        'Sua BA, entretanto, é superior. Isso reforça que o melhor previsor depende do alvo e da métrica adotados. '
        'A comparação principal permanece no horizonte previamente selecionado, e nenhum resultado é usado para escolher retroativamente outro horizonte.']
    predictive='\n\n'.join(text)+'\n'
    fragment.write_text(predictive,encoding='utf-8')
    (DEST/'predicao_xgb.tex').write_text(predictive,encoding='utf-8')
    f20=fin[fin.n.eq(20)].copy()
    prior20=float(f20[f20.layer.eq('Prior')].return_pct.iloc[0])
    f20['delta_prior']=f20.return_pct-prior20
    finance=[r'\section{Contribuição das camadas para o resultado financeiro}',r'\label{sec:atribuicao_financeira}',
        'Os scores anteriores avaliam previsões individuais. Para relacioná-los ao resultado final, '
        'as mesmas sete configurações foram inseridas no motor financeiro da rodada consolidada. '
        'Mantiveram-se datas, custos de 10 bps, janela de covariância de 126 pregões, aversão ao risco de três, '
        'restrições, calendário mensal e execução no fechamento seguinte. '
        'Somente o vetor de expectativas fornecido à seleção e à otimização foi alterado. '
        'A reexecução reproduziu os resultados anteriores do híbrido e do prior.',
        'Nas configurações XGB e Q, as expectativas entram diretamente na mesma seleção por excesso sobre volatilidade '
        'e no mesmo otimizador de média-variância, sem atualização BL. '
        'Portanto, não se trata de negociar automaticamente qualquer sinal de alta: a regra de alocação é compartilhada. '
        'Com vinte ativos, a elegibilidade é idêntica para todos, e a comparação se concentra na alocação. '
        'Com cinco ou dez, os resultados incluem também as mudanças na seleção.',
        table('Resultado financeiro das configurações com os vinte ativos.','tab:financeiro_camadas_20',
              ['Config.',r'Retorno (\%)','Sharpe',r'MDD (\%)',r'$\Delta$ vs. prior (p.p.)'],
              [[CODES[r.layer],fmt(r.return_pct),fmt(r.sharpe),fmt(r.max_drawdown_pct),fmt(r.delta_prior,4)] for r in f20.itertuples()], 'lrrrr',
              'Fonte: elaboração própria. Retornos líquidos, mesma janela de 2025 e mesmas restrições. MDD: drawdown máximo.'),
        'O uso direto de XGB ou Q no otimizador resulta em retornos negativos, enquanto as configurações que incorporam '
        'prior e atualização BL apresentam resultados positivos nesta rodada. '
        'Isso é consistente com a redução de erro observada após a atualização, mas não prova que a melhora preditiva '
        'seja a única causa da diferença financeira: a posição dos ativos no ranking, a otimização e as restrições também intervêm.',
        'No universo completo, o BL retorna 28,2085\\% e o prior sem views, 28,2053\\%: a diferença é de apenas 0,0032 p.p. '
        'O Sharpe do prior é ligeiramente maior. Assim, a vantagem financeira frente a Markowitz não pode ser atribuída integralmente '
        'ao XGBoost ou ao sentimento. O prior, a covariância e as restrições já sustentam resultado semelhante ao do sistema completo.',
        r'Fixando a cobertura e a confiança, retirar somente a polaridade textual reduz o retorno de 28,21\% para 21,43\% '
        r'($BL_0$). Essa comparação mostra contribuição favorável da polaridade dentro da configuração híbrida, '
        'mas não demonstra superioridade do híbrido frente à alternativa de não incorporar opiniões. '
        'Os efeitos condicionais das partes não são aditivos: não se podem somar diferenças de retorno para decompor o lucro total.',
        table('Sensibilidade da contribuição das opiniões na seleção própria.','tab:financeiro_camadas_selecao',
              ['n',r'Prior (\%)',r'$BL_0$ (\%)',r'BL completo (\%)',r'BL menos prior (p.p.)'],
              [[n,fmt(fin[(fin.n==n)&fin.layer.eq('Prior')].return_pct.iloc[0]),
                fmt(fin[(fin.n==n)&fin.layer.eq('BL_neutro')].return_pct.iloc[0]),
                fmt(fin[(fin.n==n)&fin.layer.eq('BL')].return_pct.iloc[0]),
                fmt(fin[(fin.n==n)&fin.layer.eq('BL')].return_pct.iloc[0]-fin[(fin.n==n)&fin.layer.eq('Prior')].return_pct.iloc[0],4)] for n in (5,10,20)],'rrrrr'),
        'Com cinco e dez ativos, o prior apresenta retornos superiores aos do BL completo. '
        'Essas cardinalidades são ilustrativas, não tamanhos escolhidos antes do teste, e suas diferenças incluem seleção e alocação. '
        'A inclusão das camadas não oferece ganho financeiro uniforme sobre o prior, embora alguns controles mostrem '
        'melhora condicional de previsão e retorno ao reintroduzir a polaridade textual.',
        'A interpretação conjunta é de um XGBoost com desempenho preditivo heterogêneo, de um componente textual com efeito dependente '
        'da forma de integração e de uma atualização BL que reduz o erro relativo das views. '
        'No resultado financeiro, porém, a principal referência estrutural continua sendo o prior. '
        'Trata-se de contribuição observada em intervenções controladas no mesmo experimento, sem evidência de significância estatística '
        'ou garantia de generalização temporal.']
    finance='\n\n'.join(finance)+'\n'
    # Integracao no capitulo fornecido, mantendo tabelas e figuras financeiras originais.
    base=base.replace('Este capítulo examina o desempenho das carteiras',
        'Este capítulo avalia a qualidade preditiva das etapas intermediárias e o desempenho das carteiras',1)
    base=base.replace(r'\caption{Desempenho das previsões utilizadas no experimento.}',
        r'\caption{Desempenho das previsões utilizadas no experimento.}'+'\n'+r'\label{tab:xgb_teste_2025}',1)
    base=base.replace('Tabela 4.2',r'Tabela~\ref{tab:xgb_teste_2025}')
    historical='No conjunto completo dos vencedores históricos, a acurácia direcional média armazenada é 55,78\\%, com R$^{2}$ negativo em 18 dos 20 ativos.'
    base=base.replace(historical,'No conjunto completo dos vencedores históricos, a acurácia direcional média armazenada é 55,78\\%. '
        'As métricas de magnitude desse resumo exigem reconciliação com as séries exportadas, discutida a seguir.')
    marker=r'\section{Comparação com ativos comuns}'
    base=base.replace(marker,r'\input{05_Resultados/predicao_xgb.tex}'+'\n\n'+marker,1)
    old='Não se compararam BL sem views, BL apenas quantitativo, BL apenas textual e híbrido completo sob um protocolo independente. Essa ablação continua necessária para atribuir ganhos a cada componente. A comparação atual identifica o desempenho do processo combinado frente a Markowitz e à Selic, mas não isola causalmente o valor incremental do XGBoost ou do sentimento.'
    new='As comparações controladas da Seção~\\ref{sec:etapas_preditivas} medem os efeitos sobre a previsão, e a Seção~\\ref{sec:atribuicao_financeira} examina sua tradução financeira. '
    new+='Esses contrastes são retrospectivos e condicionados à configuração atual. Permanece necessária uma validação independente com todas as escolhas congeladas antes do teste.'
    if old not in base: raise ValueError('Paragrafo de atribuicao da fonte nao encontrado')
    base=base.replace(old,new)
    base=base.replace(r'\section{Limitações e alcance das conclusões}',finance+'\n'+r'\section{Limitações e alcance das conclusões}',1)
    old='A terceira limitação é de atribuição. Com gate de três notícias, 6 dos 240 pares ativo-data habilitam as views. A comparação não separa os efeitos do prior, das restrições, da covariância e das fontes de inteligência computacional.'
    new='A terceira limitação é de atribuição. Com gate de três notícias, 6 dos 240 pares ativo-data habilitam as views. '
    new+='Os controles apresentados isolam intervenções específicas sobre as opiniões e a polaridade, mas não decompõem de forma aditiva '
    new+='o retorno nem separam todos os efeitos do prior, das restrições e da covariância. A quase igualdade financeira entre BL completo '
    new+='e prior sem views no universo completo exige cautela ao atribuir o ganho final às fontes de inteligência computacional.'
    base=base.replace(old,new)
    last='No universo completo, BL retorna 28,21\\%, frente aos 19,10\\% de Markowitz e aos 14,87\\% da Selic.'
    base=base.replace(last,last+' A vantagem observada frente a esses referenciais não implica superioridade preditiva uniforme '
        'ou contribuição financeira independente de cada camada, como mostram os controles do capítulo.')
    chapter.write_text(base,encoding='utf-8')
    (DEST/'resultados.tex').write_text(base,encoding='utf-8')
    (DEST/'manifest.json').write_text(json.dumps(dict(source=str(SOURCE),source_sha256=hashlib.sha256(SOURCE.read_bytes()).hexdigest(),
        financial_original_tables_preserved=True,prediction_layers=7,financial_control_runs=len(fin),
        target_files=[str(chapter),str(fragment)]),ensure_ascii=False,indent=2),encoding='utf-8')
    print('Capitulo fornecido integrado: diagnostico XGB, scores pareados e atribuicao financeira controlada.')


if __name__=='__main__':main()
