# Guia técnico do TCC

## Pergunta e arquitetura

O trabalho investiga como incorporar previsões quantitativas e sentimento de notícias à alocação de carteiras da B3. O XGBoost prevê o retorno de cada ação, as manchetes são classificadas por polaridade e os dois sinais compõem as opiniões do Black–Litterman. O posterior combina essas opiniões com o prior de equilíbrio e alimenta a otimização de pesos. [Arquitetura](../tcc/03_Arquitetura/arquitetura.tex) · [Procedimentos](../tcc/04_Procedimentos/procedimento.tex)

## Dados, datas e execução

- O universo de simulação contém 20 ações. Os dados diários têm preços de abertura, máxima, mínima e fechamento, volume e informações de mercado. O universo fixo e as revisões históricas das cotações limitam a interpretação da evidência.
- (h) é um horizonte em **pregões**. A previsão direta estima o log-retorno `log(P no pregão t+h / P no pregão t)`. O primeiro termo usa um preço ainda desconhecido no momento do sinal. Um exemplo entra no treinamento só depois que seu alvo amadurece. As janelas dos atributos, como 21 ou 63 pregões, indicam quanto histórico cada cálculo usa.
- Na carteira, o sinal é calculado no fechamento do primeiro pregão de cada mês e executado no fechamento do pregão seguinte. O retorno até o fechamento da execução pertence à carteira anterior. O backtest consolidado tem 250 pregões em 2025, janela de covariância de 126 pregões, aversão ao risco 3 e custo de 10 pontos-base sobre o giro. Os preços só ocupam datas com pregão.
- O recorte **após pausas** nos testes adicionais usa `dias_corridos > 1`. Quinta para sexta vale 1, sexta para segunda vale 3 e sexta para terça após feriado vale 4. O atributo mede a diferença entre datas presentes na série. Nos testes de 21 pregões, 2025 normalmente tem 4.580 previsões elegíveis, das quais **1.020** são de **51 datas após pausas** para **20 ações**.

[Procedimentos do TCC](../tcc/04_Procedimentos/procedimento.tex) · [Simulador financeiro](../intel/Blacklitterman/simular_dois_cenarios.py) · [Teste de calendário](../experimentos/previsao_calendario_2026_11_25/README.md)

## XGBoost oficial e variantes experimentais

No texto, o XGBoost conjunto tem horizontes de **5, 10 e 21 pregões**, com horizontes da carteira escolhidos por ação na validação de 2024. O pool oficial usa conjuntos de atributos por horizonte. No diagnóstico de 2025 para 21 pregões, o MAE foi **6,72**, o RMSE **8,42**, o acerto direcional **58,41%** e o R² médio por ação **−0,1354**. MAE e RMSE estão na escala de log-retorno × 100. O R² agregado é calculado de outra forma, reunindo observações de todas as ações. [Resultados do TCC](../tcc/05_Resultados/resultados.tex) · [Receita do pool](../analise_resultados/evolucao_xgb/receita_pool_para_confirmacao.json)

O pool experimental que apresentou **54,17% de acurácia balanceada após pausas** é outro modelo. Ele prevê diretamente o retorno de **21 pregões**, com 30 colunas na matriz `X`: oito atributos econômicos, `dias_corridos`, 20 identificadores `ticker_*` e beta de baixa. O treino aplica peso relativo de **1,04** aos retornos futuros negativos e **1,00** aos demais. O peso fica fora da matriz de entrada. O XGBoost usa erro quadrático. Os alvos são padronizados com média e desvio-padrão do trecho de treino e convertidos de volta à escala original após a previsão. Os atributos entram em `float32` na escala de suas próprias fórmulas. [Matriz experimental](../experimentos/features_baixas_2026_11_25/pesos_baixas/matriz_X_2025.csv) · [Treino ponderado](../experimentos/features_baixas_2026_11_25/pesos_baixas/treinar.py)

| Atributo | Cálculo |
|---|---|
| `eco_beta_63` | Covariância entre retornos da ação e do Ibovespa dividida pela variância do índice nos últimos 63 pregões. |
| `eco_momentum_252_21` | Log do preço de 21 pregões atrás dividido pelo preço de 252 pregões atrás. |
| `eco_relativo_ibov_21` | Soma em 21 pregões da diferença entre os log-retornos diários da ação e do Ibovespa. |
| `eco_residual_21` | Soma em 21 pregões do retorno da ação menos beta disponível no pregão anterior × retorno diário do índice. |
| `eco_vol_residual_21` | Desvio-padrão dos retornos residuais dos últimos 21 pregões. |
| `eco_vol_quedas_21` | Raiz da média de quadrados dos retornos negativos em 21 pregões, incluindo zero nos dias de alta. |
| `eco_amihud_21` | Log de 1 + 1 milhão × média em 21 pregões do retorno simples absoluto dividido por preço × volume, com mínimo de 18 observações válidas. |
| `eco_momentum_ibov_21` | Soma dos log-retornos diários do Ibovespa em 21 pregões. |
| `dias_corridos` | Diferença de calendário em dias entre o pregão atual e o anterior. |
| `ticker_*` | Identificação binária de cada uma das 20 ações. |
| `beta_baixas_63` | Beta calculado somente nos dias de queda do Ibovespa dentro de uma janela de 63 pregões, com pelo menos 15 observações elegíveis. |

O limiar zero rotula uma previsão positiva como alta e uma negativa como baixa. Calibrar o limiar altera os rótulos sem alterar os retornos previstos. A **acurácia balanceada** é a média do acerto nas altas e nas baixas, dando peso igual às duas direções. Valores realizados exatamente iguais a zero ficam fora dessa métrica. MAE e RMSE medem o erro de magnitude. O R² médio compara o erro quadrático com a média realizada de cada ação e depois agrega entre ações. A previsão zero e a média histórica conhecida na origem são referências operacionais diferentes do denominador do R².

## Notícias e Black–Litterman

- A coleta de manchetes de 2023 a 2025 reuniu **7.835 registros**. Após deduplicação e filtro de pertinência, ficaram **4.681 manchetes classificadas** em positivo, neutro e negativo. A agregação usa a empresa emissora e pode atender mais de uma classe de ação. A concordância da classificação com a referência humana foi de **73%** na avaliação reportada no TCC.
- No backtest consolidado, a janela textual usa apenas o **dia civil anterior ao sinal mensal**. A data pode ser domingo ou feriado. A polaridade agregada é `tanh(média dos escores × ln(1 + número de manchetes))`. A cobertura, a polaridade e a confiança são componentes distintos.
- O gate é de três manchetes. Abaixo do limiar, a configuração do **hard prior** reduz fortemente a confiança da opinião. Na auditoria oficial, **40 de 240** pares ação–data tinham alguma notícia e **6 de 240** atingiram o gate. A ausência de manchetes elegíveis não indica que o mercado recebeu notícias neutras.
- O cenário sem filtro de pertinência passou de **69 para 92 manchetes nos dias elegíveis** e de **6 para 12** pares acima do gate. Com 20 ações, o retorno caiu de **27,79% para 20,39%** e o Sharpe de **0,70 para 0,36**. A deduplicação isolada não mudou o conjunto nas datas usadas pelo backtest.

[Procedimentos de notícias](../tcc/04_Procedimentos/procedimento.tex) · [Resultados oficiais](../tcc/05_Resultados/resultados.tex) · [Ablação do filtro](../experimentos/filtro_noticias_2026_11_24/resultados.csv)

## Resultado financeiro consolidado de 2025

| Configuração, com 20 ações | Retorno líquido | Sharpe | Queda máxima |
|---|---:|---:|---:|
| Black–Litterman híbrido | 27,79% | 0,70 | −8,46% |
| XGBoost direto | 3,27% | −0,58 | −20,13% |
| Prior sem views | 28,21% | 0,74 | −8,46% |
| Markowitz | 19,10% | 0,29 | −12,78% |

A Selic de referência acumulou **14,87%**. Com 20 ações, o prior sem views teve a melhor combinação de retorno e Sharpe entre as quatro configurações dessa tabela. A carteira de pesos iguais também ficou em aproximadamente 28,21% no teste correspondente. Os resultados por cardinalidade dependem de seleção livre ou de subconjuntos pareados. A frequência mensal e o gate limitam a influência textual, e uma view pode afetar outras ações por meio da covariância. [Resultados oficiais](../tcc/05_Resultados/resultados.tex) · [Arquivos da simulação](../analise_resultados/simulacao_pool/)

## O que os testes de previsão mostraram

O pool experimental direto com peso 1,04 e limiar zero, no recorte após pausas de 2025, teve **MAE 6,635**, **RMSE 8,458**, **R² médio −0,132** e **54,17% de acurácia balanceada**. Acertou **80,15% das altas** e **28,19% das baixas** na média entre ações. A previsão zero teve MAE de **6,767** e RMSE de **8,554** no mesmo recorte. O ganho frente a zero foi pequeno. Também se observou amplitude prevista muito menor que a realizada em todas as ações, com **PRIO3** tendo o maior R² do diagnóstico por ativo, **+0,124**. [Erros](../experimentos/features_baixas_2026_11_25/objetivo_direcional/erros_regressao_2025.csv) · [Gráfico por ativo](../experimentos/features_baixas_2026_11_25/pesos_baixas/previsao_vs_realizado_por_ativo_2025.png)

- Oito atributos econômicos simplificaram a entrada com perda pequena de precisão no primeiro teste. A duração da pausa melhorou o acerto das duas direções em 21 pregões no comparativo inicial. O resultado variou com o conjunto de treino e com a escolha entre horizontes fixos e horizontes por ação.
- Acrescentar beta de baixa melhorou a detecção de quedas, e o peso 1,04 melhorou a acurácia balanceada. O limiar calibrado para preservar altas em 2024 reduziu demais os acertos nas baixas em 2025. A classificação logística direta, as árvores mais flexíveis, as médias históricas como atributos e os indicadores técnicos adicionais do artigo não superaram a referência ponderada em equilíbrio direcional.
- Em escala comum de retorno simples, treinar diretamente com retorno simples reduziu o RMSE de **8,752 para 8,697** em 2025 e piorou o acerto nas baixas de **28,19% para 16,50%**. A soma de 21 previsões diárias geradas na origem teve RMSE quase igual ao alvo direto, com acurácia balanceada de **52,68%**.
- O modelo de fatores com média histórica do Ibovespa teve RMSE de **8,420**, ligeiramente menor que o direto, com acerto nas baixas de **5,39%**. O modelo treinado para ranking ficou atrás do ranking produzido pelo pool atual.
- A regressão quantílica P10–P90 obteve **81,96%** de cobertura em 2025, com **7,16%** abaixo de P10 e **10,88%** acima de P90. A P50 teve MAE de **6,675**. Usar P90 nas altas e P10 nas baixas elevou o MAE para **11,600**. A interpolação pelo sinal escolheu **0%** em 2024. O ajuste pelo ranking escolheu **12%** em 2024, mas em 2025 teve MAE de **6,684**, contra 6,675 da P50.

[Catálogo dos experimentos](CATALOGO_EXPERIMENTOS.md)

## Leitura da evidência e reprodução

A validação de 2024 precede a avaliação de 2025. Várias hipóteses foram propostas depois de olhar resultados de 2025, de modo que a evidência posterior é **retrospectiva**. Os alvos de 21 pregões de datas próximas se sobrepõem. Em testes por ação e por data, as observações também compartilham fatores de mercado. Uma comparação temporal futura precisa considerar essas dependências e usar um período novo para confirmar as escolhas.

Antes de comparar números, identifique a versão do pool, o horizonte, a escala do alvo, a amostra, o método de agregação e a referência. Volte aos CSVs pareados para conferir mudanças sutis, como a restrição a uma base de treino completa nos testes de atributos originais versus simplificados. Para reproduzir, consulte o script e a auditoria de cada experimento. Os numeradores, denominadores e hashes registrados descrevem a execução histórica.

O artigo de Ran et al. usa ARIMA, LSTM, Informer ou iTransformer para **prever fatores** e entrega essas previsões ao XGBoost antes do Black–Litterman. O teste local inspirado na tabela de indicadores usou ATR, ADXR, MACD, médias móveis e RSI **observados na origem**. São hipóteses diferentes. [Teste dos fatores observados](../experimentos/fatores_artigo_2026_11_25/resumo.csv)
