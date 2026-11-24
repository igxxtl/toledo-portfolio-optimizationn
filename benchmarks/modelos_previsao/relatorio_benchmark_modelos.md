# Comparativo temporal de Lag-Llama, GRU e XGBoost para prever retornos

Novo benchmark executado com preços diários versionados no projeto `igxxtl/toledo-portfolio-optimizationn`. Serve para complementar o histórico do TCC com um confronto nas mesmas observações. Os números deste arquivo vêm da nova rodada.

[Gráfico comparativo](comparativo_benchmark.png) · [Dados de auditoria e resultados](resultados_2025.zip)

## Resultado principal

O **XGBoost do projeto acertou melhor a direção que GRU e Lag-Llama em 21 pregões**, mas ficou atrás da GRU em erro de magnitude. Para prever o valor do retorno, **a referência de retorno zero teve o menor RMSE e o menor MAE nos quatro horizontes** entre os métodos do protocolo novo. O contraste importa para o TCC porque o vetor de visões usa uma magnitude de retorno, além do sinal.

Em 21 pregões, o XGBoost obteve **57,95%** de acurácia direcional e **57,53%** de acurácia balanceada. A GRU obteve **48,49%** e **45,68%**, respectivamente. O Lag-Llama obteve **46,05%** e **50,69%**. Dos retornos realizados, **58,03%** foram positivos. A vantagem do XGBoost sobre as duas redes nesse horizonte aparece sobretudo quando se distingue a capacidade de acertar altas e baixas da frequência bruta de altas.

O **ensemble XGBoost–LightGBM** teve a maior acurácia direcional observada em 21 pregões (**58,30%**) e a maior acurácia balanceada (**58,29%**). Sua diferença de acurácia direcional frente ao XGBoost do projeto foi de **0,35 p.p.**, com intervalo de 95% de **−1,22 a 2,07 p.p.** no bootstrap de blocos de 21 datas. Isso deixa em aberto se a vantagem observada se repetiria.

## 1. Pergunta e protocolo

**Pergunta.** Para prever os retornos futuros usados como sinal quantitativo no Black–Litterman, como se comparam o XGBoost do projeto, a GRU, o Lag-Llama e referências simples quando se mantêm fixos o alvo e a amostra?

| Elemento | Definição |
|---|---|
| Base | CSVs diários do commit `e07196bc76da0444f4a5f0aeefcb1b20bfd2776a`. Usam-se os 20 ativos do projeto, com início de histórico em 2018 quando disponível. |
| Ativos | ABEV3, B3SA3, BBAS3, BBDC4, CMIG4, EQTL3, GGBR4, ITUB4, LREN3, MGLU3, PETR3, PETR4, PRIO3, RADL3, RAIL3, SANB11, SBSP3, SUZB3, VALE3 e WEGE3. |
| Alvo | Log-retorno de fechamento `ln(C[t+h]/C[t])` do ativo. A previsão pode usar dados até o fechamento de origem. |
| Horizontes | 1, 5, 10 e 21 pregões. O horizonte de um pregão acrescenta uma referência diária. Os demais coincidem com o estudo de horizontes anterior. |
| Validação do XGBoost | 2024, com treino anterior a 2024. Escolha por ativo e horizonte baseada em acurácia de sinal e desempate por RMSE. As escolhas ficam congeladas antes do teste. |
| Teste | 2025, com alvo realizado até o último pregão do próprio ano. Atualização em janela expansiva no início de cada trimestre. O treino usa somente alvos realizados antes do início do trimestre. Os retornos já realizados em trimestres anteriores podem entrar naquele treino. |
| Início útil do treino | Descarte das primeiras 252 posições de cada série para a janela longa de momentum. A GRU tem janela adicional de 35 posições na montagem das sequências. |
| Comparação | Exatamente os mesmos pares ativo–data–horizonte e o mesmo retorno realizado para todos os métodos. Ausências, chaves duplicadas e valores não finitos são bloqueados na consolidação. |
| Escopo | Sinal quantitativo de retornos. Notícias, formação de carteiras, custos de negociação e Black–Litterman ficam fora da medição. |

[Snapshot dos preços do projeto](https://github.com/igxxtl/toledo-portfolio-optimizationn/tree/e07196bc76da0444f4a5f0aeefcb1b20bfd2776a/dados_diarios) · [Pipeline XGBoost original](https://github.com/igxxtl/toledo-portfolio-optimizationn/blob/e07196bc76da0444f4a5f0aeefcb1b20bfd2776a/intel/model_training/train_xgb.py)

## 2. Modelos e referências

| Método | Entrada e estimação |
|---|---|
| **XGBoost regressão** | Reusa os construtores de atributos do repositório, incluindo variáveis de preço, volume, tendência, risco, Ibovespa, indicadores macro disponíveis e ranks transversais. Ajuste separado por ativo e horizonte. Objetivo de erro quadrático e pesos exponenciais crescentes no tempo. |
| **XGBoost 3 classes** | Mesmos atributos. Tercis do alvo do treino definem as classes. A previsão numérica é a soma das médias de retorno por classe ponderadas pelas probabilidades previstas. |
| **XGBoost projeto** | Adaptação temporal do nosso pipeline. Para cada ativo e horizonte, seleciona em 2024 uma das duas variantes acima. A busca usa três configurações de profundidade/número de árvores (2/120, 3/220, 5/300) por variante. Subamostragem de linhas e colunas 0,8, taxa de aprendizado 0,035, peso mínimo de filho 12, L2 igual a 5 e decaimento amostral `exp(linspace(-3,0,N))`. A busca é nova e os artefatos históricos ajustados após o período de teste ficam fora do ajuste. |
| **GRU** | Rede recorrente com 35 pregões dos mesmos atributos tabulares, camada GRU de 16 unidades, embedding por ativo com quatro dimensões e cabeça de regressão 20→32→1 com dropout 0,30. Treino conjunto nos 20 ativos, separado por horizonte e trimestre. Ajuste da imputação por mediana e da padronização por ativo no treino. AdamW, taxa de aprendizado 0,0005, até 18 épocas, lote 512. Os últimos 15% do treino de cada série formam a validação interna de parada antecipada, com separação temporal dos rótulos. Restaura a época de menor MSE interno. |
| **Lag-Llama** | Checkpoint público pré-treinado, em modo *zero-shot*. O hash SHA-256 do arquivo histórico do repositório coincide com o do checkpoint público. Entrada univariada de log-retornos diários observados até a origem com sequência de até 1.200 pontos, contexto do modelo 32 e inferência de até 21 passos. A previsão pontual realimenta a média condicional Student-t e acumula os passos até cada horizonte. O GluonTS indexa a série numa frequência de dias úteis, tratando a sequência de pregões como regular, sem um calendário específico de feriados da B3. Sem ajuste supervisionado na B3 ou uso de notícias. |
| **LightGBM** | Regressor de boosting tabular por ativo e horizonte, com 150 árvores, profundidade 3, até sete folhas e parâmetros fixos. Usa a mesma janela expansiva, atributos e ponderação temporal. |
| **Ensemble XGB–LGBM** | Média aritmética das previsões de XGBoost regressão e LightGBM. Testa a ideia do ensemble opcional que aparecia nas versões históricas do projeto. |
| **Ridge** | Regressão linear regularizada com os mesmos atributos tabulares e treinamento por ativo e horizonte, penalidade 100, mediana para imputação e padronização ajustadas no treino. |
| **Média histórica** | Retorno médio dos rótulos já realizados no treino de cada ativo e horizonte. |
| **Zero** | Prevê retorno nulo, equivalente a manter o fechamento de origem como previsão de preço. A direção fica sem avaliação para essa referência. |

A GRU histórica classificava excesso de retorno sobre o Ibovespa. A GRU deste benchmark tem saída de retorno futuro para comparar diretamente com o vetor quantitativo do projeto. O Lag-Llama histórico operava sobre VWAP intradiário. Aqui se testa sua transferência para retorno de fechamento diário. XGBoost e GRU recebem dados tabulares adicionais e o Lag-Llama recebe o histórico da própria série. Assim, o resultado avalia as configurações completas usadas neste protocolo, incluindo seus conjuntos de informação e regimes de estimação.

[Checkpoint oficial e SHA-256](https://huggingface.co/time-series-foundation-models/Lag-Llama/blob/main/lag-llama.ckpt) · [Código do Lag-Llama na revisão executada](https://github.com/time-series-foundation-models/lag-llama/tree/df7531a83a19b3c6a0222d703ca9bf59ef7a6ab9)

## 3. Resultados detalhados

As métricas agregadas dão o mesmo peso a cada observação pareada. Direção é a proporção de sinais exatamente iguais entre previsão e realização. A acurácia balanceada dá peso igual às classes retorno positivo e retorno não positivo. AUC avalia a ordenação da classe positiva pelo retorno previsto, sem fixar o limiar. RMSE e MAE são reportados na escala de log-retorno multiplicada por 100. O \(R^2\) é calculado em torno da média observada do grupo e pode ser negativo. O pacote inclui ainda correlação de Pearson, viés e frequência de positivos.

### 1 pregão

4.980 pares ativo-data em 249 datas, com origens de 02/01/2025 a 29/12/2025. A proporção de retornos positivos foi 51,45%.

| Modelo | RMSE (%) | MAE (%) | Direção (%) | Bal. (%) | AUC | R² |
|---|---:|---:|---:|---:|---:|---:|
| Zero | 1,986 | 1,354 | — | — | — | -0,002 |
| Media historica | 1,988 | 1,355 | 49,68 | 49,06 | 0,488 | -0,003 |
| GRU | 1,993 | 1,364 | 49,84 | 49,81 | 0,492 | -0,009 |
| XGBoost 3 classes | 2,023 | 1,388 | 49,70 | 50,17 | 0,503 | -0,039 |
| LightGBM | 2,038 | 1,394 | 50,64 | 50,88 | 0,509 | -0,054 |
| Ensemble XGB-LGBM | 2,046 | 1,399 | 50,14 | 50,43 | 0,508 | -0,063 |
| XGBoost projeto | 2,056 | 1,405 | 49,18 | 49,69 | 0,499 | -0,073 |
| Ridge | 2,063 | 1,420 | 50,70 | 50,92 | 0,509 | -0,080 |
| XGBoost regressao | 2,065 | 1,413 | 50,06 | 50,41 | 0,507 | -0,082 |
| Lag-Llama | 2,066 | 1,419 | 48,33 | 48,97 | 0,485 | -0,084 |

### 5 pregões

4.900 pares ativo-data em 245 datas, com origens de 02/01/2025 a 19/12/2025. A proporção de retornos positivos foi 54,16%.

| Modelo | RMSE (%) | MAE (%) | Direção (%) | Bal. (%) | AUC | R² |
|---|---:|---:|---:|---:|---:|---:|
| Zero | 4,327 | 3,114 | — | — | — | -0,008 |
| Media historica | 4,346 | 3,122 | 51,02 | 49,24 | 0,484 | -0,017 |
| GRU | 4,404 | 3,181 | 49,02 | 48,17 | 0,470 | -0,044 |
| Lag-Llama | 4,603 | 3,341 | 48,82 | 50,08 | 0,502 | -0,140 |
| XGBoost 3 classes | 4,616 | 3,308 | 49,37 | 50,04 | 0,508 | -0,147 |
| XGBoost projeto | 4,635 | 3,350 | 49,24 | 49,93 | 0,508 | -0,156 |
| Ensemble XGB-LGBM | 4,657 | 3,360 | 50,27 | 50,77 | 0,514 | -0,167 |
| LightGBM | 4,671 | 3,357 | 50,61 | 51,04 | 0,510 | -0,174 |
| XGBoost regressao | 4,678 | 3,389 | 50,41 | 51,00 | 0,516 | -0,178 |
| Ridge | 4,707 | 3,382 | 51,76 | 51,42 | 0,517 | -0,193 |

### 10 pregões

4.800 pares ativo-data em 240 datas, com origens de 02/01/2025 a 12/12/2025. A proporção de retornos positivos foi 56,31%.

| Modelo | RMSE (%) | MAE (%) | Direção (%) | Bal. (%) | AUC | R² |
|---|---:|---:|---:|---:|---:|---:|
| Zero | 6,234 | 4,588 | — | — | — | -0,015 |
| Media historica | 6,290 | 4,605 | 51,17 | 48,31 | 0,470 | -0,034 |
| GRU | 6,429 | 4,734 | 49,27 | 47,86 | 0,452 | -0,080 |
| XGBoost 3 classes | 6,645 | 4,847 | 50,94 | 51,64 | 0,523 | -0,153 |
| XGBoost projeto | 6,846 | 5,003 | 51,31 | 51,94 | 0,522 | -0,224 |
| Ridge | 6,850 | 4,981 | 53,29 | 52,47 | 0,530 | -0,226 |
| Lag-Llama | 6,852 | 5,094 | 48,27 | 50,77 | 0,509 | -0,227 |
| LightGBM | 6,877 | 4,978 | 52,19 | 52,47 | 0,527 | -0,235 |
| Ensemble XGB-LGBM | 6,880 | 4,999 | 52,15 | 52,36 | 0,525 | -0,236 |
| XGBoost regressao | 6,927 | 5,050 | 52,08 | 52,34 | 0,522 | -0,253 |

### 21 pregões

4.580 pares ativo-data em 229 datas, com origens de 02/01/2025 a 27/11/2025. A proporção de retornos positivos foi 58,03%.

| Modelo | RMSE (%) | MAE (%) | Direção (%) | Bal. (%) | AUC | R² |
|---|---:|---:|---:|---:|---:|---:|
| Zero | 9,187 | 6,813 | — | — | — | -0,031 |
| Media historica | 9,371 | 6,891 | 52,14 | 48,22 | 0,454 | -0,073 |
| GRU | 9,718 | 7,196 | 48,49 | 45,68 | 0,426 | -0,154 |
| XGBoost 3 classes | 10,259 | 7,261 | 55,85 | 55,61 | 0,581 | -0,286 |
| Ensemble XGB-LGBM | 10,293 | 7,198 | 58,30 | 58,29 | 0,599 | -0,295 |
| XGBoost regressao | 10,311 | 7,229 | 58,03 | 57,79 | 0,596 | -0,299 |
| LightGBM | 10,349 | 7,230 | 57,71 | 58,08 | 0,600 | -0,309 |
| Ridge | 10,370 | 7,518 | 56,22 | 54,61 | 0,569 | -0,314 |
| XGBoost projeto | 10,460 | 7,382 | 57,95 | 57,53 | 0,580 | -0,337 |
| Lag-Llama | 11,074 | 8,225 | 46,05 | 50,69 | 0,509 | -0,498 |

### Diferenças pareadas contra XGBoost projeto

| Pregões | Modelo | Δ RMSE (p.p.) e IC 95% | Δ MAE (p.p.) e IC 95% | Δ direção (p.p.) e IC 95% |
|---:|---|---|---|---|
| 1 | Ensemble XGB-LGBM | -0,010 [-0,018 a -0,002] | -0,006 [-0,013 a 0,001] | 0,964 [-0,502 a 2,410] |
| 1 | GRU | -0,063 [-0,083 a -0,043] | -0,041 [-0,055 a -0,026] | 0,663 [-1,606 a 2,871] |
| 1 | Lag-Llama | 0,010 [-0,022 a 0,041] | 0,014 [-0,009 a 0,035] | -0,843 [-3,112 a 1,546] |
| 1 | LightGBM | -0,018 [-0,030 a -0,007] | -0,011 [-0,020 a -0,002] | 1,466 [0,060 a 2,771] |
| 1 | Media historica | -0,068 [-0,087 a -0,050] | -0,050 [-0,062 a -0,038] | 0,502 [-2,129 a 3,113] |
| 1 | Ridge | 0,006 [-0,027 a 0,035] | 0,015 [-0,011 a 0,039] | 1,526 [-0,884 a 4,017] |
| 1 | Zero | -0,070 [-0,091 a -0,051] | -0,051 [-0,063 a -0,040] | — |
| 5 | Ensemble XGB-LGBM | 0,022 [-0,040 a 0,083] | 0,010 [-0,030 a 0,053] | 1,020 [-0,408 a 2,633] |
| 5 | GRU | -0,231 [-0,357 a -0,120] | -0,169 [-0,265 a -0,077] | -0,224 [-4,429 a 3,919] |
| 5 | Lag-Llama | -0,032 [-0,202 a 0,143] | -0,009 [-0,128 a 0,112] | -0,429 [-4,367 a 3,633] |
| 5 | LightGBM | 0,036 [-0,046 a 0,114] | 0,008 [-0,036 a 0,054] | 1,367 [-0,245 a 2,980] |
| 5 | Media historica | -0,289 [-0,409 a -0,175] | -0,228 [-0,310 a -0,146] | 1,776 [-2,816 a 6,408] |
| 5 | Ridge | 0,073 [-0,104 a 0,227] | 0,033 [-0,124 a 0,174] | 2,510 [-1,286 a 6,898] |
| 5 | Zero | -0,308 [-0,430 a -0,196] | -0,236 [-0,317 a -0,159] | — |
| 10 | Ensemble XGB-LGBM | 0,034 [-0,083 a 0,139] | -0,004 [-0,083 a 0,068] | 0,833 [-0,312 a 2,062] |
| 10 | GRU | -0,417 [-0,689 a -0,172] | -0,269 [-0,504 a -0,047] | -2,042 [-7,251 a 3,105] |
| 10 | Lag-Llama | 0,007 [-0,316 a 0,329] | 0,091 [-0,190 a 0,370] | -3,042 [-8,188 a 2,021] |
| 10 | LightGBM | 0,031 [-0,109 a 0,166] | -0,025 [-0,130 a 0,070] | 0,875 [-0,562 a 2,396] |
| 10 | Media historica | -0,556 [-0,826 a -0,313] | -0,398 [-0,626 a -0,183] | -0,146 [-5,480 a 5,104] |
| 10 | Ridge | 0,004 [-0,389 a 0,344] | -0,022 [-0,369 a 0,285] | 1,979 [-2,688 a 7,146] |
| 10 | Zero | -0,612 [-0,870 a -0,364] | -0,415 [-0,620 a -0,211] | — |
| 21 | Ensemble XGB-LGBM | -0,167 [-0,315 a -0,005] | -0,184 [-0,265 a -0,110] | 0,349 [-1,223 a 2,074] |
| 21 | GRU | -0,743 [-1,701 a 0,157] | -0,185 [-0,979 a 0,501] | -9,454 [-16,813 a -1,048] |
| 21 | Lag-Llama | 0,614 [-0,268 a 1,535] | 0,844 [0,105 a 1,576] | -11,900 [-18,755 a -4,825] |
| 21 | LightGBM | -0,111 [-0,303 a 0,101] | -0,152 [-0,245 a -0,053] | -0,240 [-2,205 a 1,703] |
| 21 | Media historica | -1,089 [-2,030 a -0,222] | -0,491 [-1,246 a 0,175] | -5,808 [-13,493 a 2,948] |
| 21 | Ridge | -0,090 [-1,274 a 1,024] | 0,136 [-0,839 a 0,979] | -1,725 [-10,459 a 7,752] |
| 21 | Zero | -1,273 [-2,144 a -0,422] | -0,568 [-1,220 a 0,014] | — |

### Diagnóstico adicional do XGBoost arquivado no projeto

O recorte usa os CSVs originais e valida data, ativo, horizonte e igualdade do retorno realizado. A trilha arquivada usa o protocolo original e fica fora do ranking e dos intervalos do novo benchmark.

| Pregões | Observações | RMSE (%) | MAE (%) | Direção (%) |
|---:|---:|---:|---:|---:|
| 5 | 4900 | 4,456 | 3,197 | 50,67 |
| 10 | 4800 | 6,388 | 4,654 | 50,27 |
| 21 | 4580 | 9,826 | 7,108 | 54,65 |

Para os intervalos, a diferença é **modelo da linha menos XGBoost projeto**. Erro menor corresponde a diferença negativa. Direção melhor corresponde a diferença positiva. O bootstrap circular usa 4.000 reamostragens de blocos consecutivos de datas e mantém juntos os ativos de cada data, preservando dependência transversal e parte da autocorrelação criada por alvos sobrepostos. A tabela usa blocos de `max(10,h)` origens e o CSV contém também a sensibilidade com blocos de `max(20,2h)`. Os intervalos são exploratórios, pontuais e sem correção por comparações múltiplas.

## 4. Leitura para o TCC

- **Magnitude.** A GRU teve RMSE e MAE menores que o XGBoost selecionado nos quatro horizontes. As diferenças de RMSE excluíram zero em 1, 5 e 10 pregões no bootstrap principal. Em 21 pregões, o RMSE foi de 9,718% na GRU, 10,460% no XGBoost selecionado e 9,187% na previsão zero. O XGBoost teve RMSE maior que o da referência de retorno zero em 1, 5, 10 e 21 pregões, com intervalos pareados que excluíram zero nos quatro casos.
- **Direção.** Em 21 pregões, a diferença de GRU menos XGBoost foi de −9,45 p.p. (IC 95% de −16,81 a −1,05), e a de Lag-Llama menos XGBoost foi de −11,90 p.p. (IC 95% de −18,76 a −4,83). Os dois intervalos continuaram abaixo de zero com blocos de 42 datas. Em 1, 5 e 10 pregões, os intervalos de diferença direcional de ambos incluíram zero no bootstrap principal. Nos horizontes de 5 e 10, a melhor acurácia direcional entre os métodos testados foi da Ridge, ainda abaixo da frequência bruta de altas.
- **Variação ao longo do ano.** Para 21 pregões, a acurácia balanceada do XGBoost do projeto ficou entre 56,02% e 61,09% nos quatro trimestres. O ensemble ultrapassou o XGBoost nos dois primeiros e ficou abaixo nos dois últimos. A frequência de altas nesse horizonte variou de 51,82% no terceiro trimestre a 74,51% no quarto. A acurácia bruta isolada pode esconder essa mudança na distribuição.
- **Artefatos originais.** Em 2025, as predições arquivadas de 21 pregões atingiram 54,65% de acurácia direcional e RMSE de 9,826%, enquanto o pipeline com seleção congelada alcançou 57,95% e 10,460%. São dois protocolos aplicados ao mesmo conjunto de retornos realizados. A evidência do arquivo histórico cobre os três horizontes disponíveis e ajuda a interpretar a reestimação.
- **Redação acadêmica.** Os resultados dão respaldo empírico ao uso do XGBoost como sinal direcional de 21 pregões em 2025 perante as implementações de GRU e Lag-Llama testadas. A estimativa da magnitude permaneceu inferior à referência de retorno zero. A conclusão deve especificar a tarefa, o horizonte, o ano e o critério de desempenho.

O benchmark usa um único ano final e um universo retrospectivo fixo. Pode haver viés de seleção ou sobrevivência em relação ao universo que estaria disponível em tempo real. Os preços vêm de um snapshot e podem incorporar revisões ou ajustes de preços conhecidos depois da data de origem. O conjunto de preditores tem informações diferentes entre famílias, assim como os respectivos orçamentos de ajuste. O experimento informa a escolha do sinal quantitativo dentro deste desenho e seria preciso testar outros períodos e medir o efeito no portfólio para fundamentar uma conclusão mais ampla. Os resultados de 2025 não explicam retrospectivamente qual foi a motivação da decisão original.

## 5. Auditoria e reprodução

- `previsoes_pareadas_2025.csv` guarda cada previsão por data de origem, data de realização, ativo, horizonte e modelo. `validacao_xgboost_2024.csv` guarda as duas variantes selecionadas na validação, e `selecao_tabular.json` registra a configuração escolhida e sua pontuação.
- `metricas_agregadas.csv`, `metricas_por_ativo.csv` e `metricas_por_trimestre.csv` permitem verificar o desempenho em cada recorte. `intervalos_vs_xgboost.csv` contém as duas sensibilidades por tamanho de bloco, e os CSVs com `xgboost_arquivado` reproduzem as métricas do diagnóstico histórico. O arquivo `lag_checkpoint.json` registra o hash e a configuração recuperada.
- Ambiente usado. Python 3.11.9, NumPy 2.4.0, pandas 2.3.3, SciPy 1.17.1, scikit-learn 1.8.0, XGBoost 3.2.0, PyTorch 2.15.0 CPU, LightGBM 4.6.0, GluonTS 0.14.4 e Lightning 2.5.6. Para compatibilidade com GluonTS, a execução Lag-Llama usou NumPy 1.26.4 e pandas 2.1.4. Os scripts têm semente 42. Em ABEV3, conferi que uma execução separada reproduziu as previsões do XGBoost e do Lag-Llama. O código do projeto de origem e a implementação do Lag-Llama devem ser fixados aos commits indicados.
- Para reproduzir, use Python 3.11 e execute os comandos abaixo na pasta deste relatório. As variáveis e caminhos do exemplo usam PowerShell. O primeiro ambiente roda os modelos tabulares, a GRU e a análise. O segundo usa as versões de NumPy e pandas compatíveis com a execução original do Lag-Llama.

```powershell
git clone https://github.com/igxxtl/toledo-portfolio-optimizationn.git repo
git -C repo switch --detach e07196bc76da0444f4a5f0aeefcb1b20bfd2776a
git -C repo restore --source=c0bbbe37b015b3bc2fe2939074f894f52928a29b -- checkpoints/lag-llama.ckpt
git clone https://github.com/time-series-foundation-models/lag-llama.git lag-llama
git -C lag-llama switch --detach df7531a83a19b3c6a0222d703ca9bf59ef7a6ab9

python -m venv .venv
python -m venv .venv-lag
$tab = '.venv/Scripts/python.exe'
$lag = '.venv-lag/Scripts/python.exe'
& $tab -m pip install torch==2.15.0 --index-url https://download.pytorch.org/whl/cpu
& $tab -m pip install numpy==2.4.0 pandas==2.3.3 scipy==1.17.1 scikit-learn==1.8.0 xgboost==3.2.0 lightgbm==4.6.0 optuna==4.8.0 matplotlib==3.10.9 joblib==1.5.3
& $lag -m pip install torch==2.15.0 --index-url https://download.pytorch.org/whl/cpu
& $lag -m pip install numpy==1.26.4 pandas==2.1.4 scipy==1.17.1 lightning==2.5.6 pytorch-lightning==2.5.6 torchmetrics==1.7.4 einops==0.8.2 pydantic==2.12.5 tqdm==4.67.1 toolz==1.2.0
& $lag -m pip install --no-deps gluonts==0.14.4

& $tab benchmark_modelos.py --repo repo --out execucao --engine tree
& $tab benchmark_modelos.py --repo repo --out execucao --engine gru
& $tab benchmark_modelos.py --repo repo --out execucao --engine lightgbm
& $lag benchmark_modelos.py --repo repo --lag-source lag-llama --out execucao --engine lag
& $tab -m pip freeze | Set-Content -Encoding utf8 execucao/ambiente.txt
& $lag -m pip freeze | Add-Content -Encoding utf8 execucao/ambiente.txt
& $tab analisar_benchmark_modelos.py --inputs execucao --out execucao --repo repo
```
