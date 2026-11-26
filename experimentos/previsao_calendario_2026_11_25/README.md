# Duração da pausa entre pregões no XGBoost

Experimento de 25 de novembro de 2026. O objetivo foi testar se informar ao XGBoost quantos dias corridos se passaram desde o último pregão melhora a previsão de retorno. A comparação usa o modelo conjunto com oito atributos econômicos como referência e altera apenas os atributos de entrada. Os resultados se referem à previsão, com foco em 2025.

## Como a variável foi construída

Para cada ação, ordenamos as observações por data de pregão. A variável `dias_corridos` é a diferença, em dias de calendário, entre a data do pregão atual e a data do pregão imediatamente anterior na série da ação.

| Pregão anterior | Pregão atual | `dias_corridos` |
|---|---|---:|
| Segunda-feira | Terça-feira | 1 |
| Sexta-feira | Segunda-feira | 3 |
| Sexta-feira | Terça-feira, com feriado na segunda | 4 |

A variável está disponível na data do sinal. Os preços continuam indexados por pregão, e os alvos continuam sendo o log-retorno acumulado nos 5, 10 ou 21 pregões seguintes. Como a diferença é calculada entre observações da série, uma cotação ausente também pode aumentar o intervalo medido.

## O que foi comparado

| Variante | Entradas além dos oito atributos econômicos |
|---|---|
| `enxuto` | Nenhuma |
| `pausa` | `dias_corridos` |
| `salto` | `gap_abertura`, o logaritmo da abertura atual dividida pelo fechamento anterior |
| `pausa_e_salto` | `dias_corridos` e `gap_abertura` |

Usamos as mesmas 20 ações, datas, alvos, identidade das ações e configuração do XGBoost para os quatro modelos. A seleção dos atributos de referência foi feita com as linhas de treinamento anteriores à validação interna. Os novos atributos foram acrescentados conforme cada variante da tabela. A validação foi semestral em 2024, e a avaliação retrospectiva foi trimestral em 2025. O treinamento de cada bloco usou somente alvos maturados antes do seu início, com uma separação adicional antes da validação interna que define o número de árvores.

O recorte `apos_pausa` considera apenas previsões feitas quando `dias_corridos > 1`. As taxas de acerto nas altas e nas baixas usam limiar de retorno zero e são calculadas separadamente sobre altas e baixas realizadas. A acurácia balanceada é a média das duas taxas. Retornos realizados iguais a zero ficam fora dessa métrica. Os indicadores das tabelas são médias entre as 20 ações.

## Resultado de 2025

Cada célula compara **modelo enxuto → modelo enxuto com duração da pausa**. RMSE está na escala de log-retorno multiplicado por 100.

| Horizonte | RMSE | Acerto direcional | Acerto nas altas | Acerto nas baixas | Acurácia balanceada |
|---|---:|---:|---:|---:|---:|
| 5 pregões | 4,114 → 4,101 | 53,39% → 54,12% | 81,57% → 87,68% | 18,61% → 14,13% | 50,09% → 50,90% |
| 10 pregões | 5,900 → 5,832 | 53,63% → 56,19% | 80,76% → 96,06% | 17,12% → 4,08% | 48,94% → 50,07% |
| 21 pregões | 8,499 → 8,450 | 56,07% → 59,04% | 76,56% → 79,51% | 24,39% → 26,87% | 50,47% → 53,19% |

No primeiro pregão após pausas, a variante `pausa` reduziu o RMSE em **1,34%** no horizonte de 10 pregões. Nesse recorte, o acerto bruto subiu de **55,47% para 58,49%**, enquanto o acerto nas baixas caiu de **19,37% para 4,63%**. Em 21 pregões, a acurácia balanceada após pausas subiu de **49,93% para 52,37%**, com melhora nas duas direções.

O salto de abertura sozinho teve o menor RMSE em 5 pregões por uma margem pequena. A duração da pausa sozinha teve o menor RMSE em 10 e 21 pregões. Na amostra completa da validação de 2024, a duração da pausa reduziu o RMSE apenas em 21 pregões. **O horizonte de 21 pregões é o candidato para confirmação em outro período**, pois em 2025 melhorou tanto o erro quanto o acerto nas altas e nas baixas. A comparação de 2025 é retrospectiva.

## Reprodução e arquivos

Execute o comando abaixo na raiz do TCC com o ambiente Python do repositório `toledo-portfolio-optimizationn`. O script usa a estrutura local em que o TCC está em `Documents/TCC 2` e o código do modelo está em `Documents/GitHub/toledo-portfolio-optimizationn`. A execução original usou Python 3.11.9 e XGBoost 3.2.0.

```powershell
& '..\GitHub\toledo-portfolio-optimizationn\venv\Scripts\python.exe' -B -X utf8 `
  'experimentos\previsao_calendario_2026_11_25\testar_calendario.py'
```

O script reproduziu **28.110 previsões** do modelo enxuto de referência e conferiu os hashes de **135 arquivos** antes e depois do teste. Ele gera os três primeiros CSVs abaixo. Os dois CSVs de acerto por direção foram calculados posteriormente a partir das previsões salvas, e seu acerto direcional bruto foi conferido contra o resumo do script.

- [Previsões pareadas](previsoes.csv)
- [Resumo por horizonte, variante e recorte](resumo.csv)
- [Métricas preditivas por ação](metricas_por_acao.csv)
- [Acerto por direção e horizonte](acerto_por_direcao.csv)
- [Acerto por direção e ação](acerto_por_direcao_e_acao.csv)
- [Previsões do experimento econômico usado como referência](../features_economicas_2026_11_25/previsoes.csv)
