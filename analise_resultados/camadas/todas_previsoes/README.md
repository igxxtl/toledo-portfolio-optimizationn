# Todas as previsões e suas relações

Entrada principal de reprodução: `python -m analise_resultados.todas_previsoes` (numpy, pandas e matplotlib). O comando atualiza **o mesmo** fragmento `Documents/TCC 2/05_Resultados/predicao_xgb.tex`. O script anterior `avaliar_predicao_xgb` produz apenas o recorte de 2025 e pode substituir esse fragmento se executado depois; para a análise completa, rode `todas_previsoes` por último.

## Cobertura

60 arquivos por ativo/horizonte, 41.852 previsões OOF, 2023-02-14 a 2026-03-31. Os vinte aliases dos vencedores não são contados novamente. Não há treino novo, ampliação retroativa do backtest ou alegação de teste temporal independente. As métricas nativas usam `y_real` e `y_pred` salvos.

`xgb_60_modelos.csv` contém N, MAE, MSE, RMSE, acurácia direcional, R², Pearson previsão/alvo, NRMSE, viés, desvios e razão de amplitude por ativo/horizonte. `xgb_todas_observacoes.csv` contém cada par individual com erros e acerto direcional. `xgb_por_ano.csv` e `xgb_por_fold.csv` preservam os recortes; folds são específicos de cada modelo, não janelas idênticas entre todos.

`reconciliacao_metricas_antigas.csv` identifica 25 casos em que as métricas de magnitude históricas descrevem o regressor, mas a série exportada é do classificador escolhido. Os valores apresentados foram recalculados das previsões efetivamente salvas. Os arquivos antigos não foram alterados. As duas séries individuais concorrentes de cada combinação não estão disponíveis: as acurácias regressor/classificador no resumo de treinamento são preservadas, mas não permitem reconstruir seus erros de magnitude separadamente.

## Comparação entre camadas

`camadas_pares.csv`: XGB, Q, prior e posterior nas mesmas decisões mensais de 2025; 228 pares elegíveis de 240, com 12 excluídos por desfecho além de 30/12/2025. `camadas_exclusoes.csv` documenta as exclusões.

`camadas_metricas.csv`: comparação conjunta e por ativo, com dois alvos distintos: excesso de log-retorno diário equivalente ao horizonte selecionado e excesso de retorno simples no fechamento seguinte. A escala do XGB/Q e a escala aritmética do prior/BL são compatibilizadas aproximadamente pelo motor; o texto explicita essa limitação. O resultado não equivale a um backtest, e o alvo seguinte não é o alvo de treinamento do XGB.

`camadas_transformacoes.csv` mede mudança média, concordância de sinal e correlação entre previsões. `camadas_correlacoes.csv` não mede correlação com resultados realizados: isso aparece em `camadas_metricas.csv`. O CSV de Q/Omega do pipeline diário antigo contém bytes inválidos, inclusive após seu cabeçalho; não é reparado silenciosamente nem misturado com a rodada mensal atual.

## Relações entre métricas

RMSE >= MAE na mesma amostra. NRMSE = sqrt(1-R²) quando o denominador é o desvio populacional do alvo. Acurácia ignora magnitude. Pearson ignora calibração de nível e amplitude: pode ser positivo mesmo com R² negativo. As correlações entre métricas estão em `relacoes_metricas_pearson.csv` e `relacoes_metricas_spearman.csv`, calculadas sobre 60 combinações dependentes, sem testes de significância iid.

As figuras foram inspecionadas visualmente. A compilação do projeto LaTeX completo foi tentada no editor. Foram adicionadas dependências ausentes (`arrows.meta`, `patterns`, `amssymb`) ao preâmbulo; a compilação permanece interrompida por outro erro preexistente de TikZ (`drop shadow` na arquitetura). Os dados e o fragmento foram preservados.
