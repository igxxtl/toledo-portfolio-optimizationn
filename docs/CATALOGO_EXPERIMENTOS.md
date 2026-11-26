# Catálogo dos experimentos do TCC

As entradas abaixo apontam para resultados preservados com os scripts, dados intermediários e protocolos disponíveis. Para a metodologia e os resultados que constam no texto, consulte o [guia técnico](GUIA_TECNICO.md) e o [capítulo de resultados](../tcc/05_Resultados/resultados.tex).

## Alocação, notícias e sensibilidade

| Teste | Evidência principal |
|---|---|
| XGBoost oficial, alocação e referências | [Simulação do pool](../analise_resultados/simulacao_pool/) |
| Peso igual, Markowitz e Black–Litterman por cardinalidade | [Resumo por tamanho](../experimentos/pesos_iguais_2026_11_20/resumo_por_tamanho.csv) |
| Sensibilidade ao hard prior | [Relatório](../experimentos/hard_prior_2026_11_20/RELATORIO.txt) e [carteira com 20 ações](../experimentos/hard_prior_2026_11_20/comparacao_20_acoes.csv) |
| Remoção de deduplicação e filtro de pertinência | [Resultados](../experimentos/filtro_noticias_2026_11_24/resultados.csv) e [cobertura](../experimentos/filtro_noticias_2026_11_24/cobertura.csv) |
| Pares ação–data sem notícias | [Levantamento complementar](../experimentos/noticias_pares_ausentes_2026_11_25/) |
| Teste com base complementar de notícias | [Pasta da execução](../experimentos/backtest_base_noticias_2026_11_25/) |
| Polaridade, avaliação humana e camadas intermediárias | [Diagnósticos das camadas](../analise_resultados/camadas/) |

## Atributos e previsão pontual

| Teste | Evidência principal |
|---|---|
| Evolução do XGBoost no repositório de modelos | [Relatório histórico](../analise_resultados/evolucao_xgb/RELATORIO.txt) |
| Oito atributos econômicos | [Resumo de previsão](../experimentos/features_economicas_2026_11_25/resumo.csv) e [carteiras](../experimentos/features_economicas_2026_11_25/carteiras.csv) |
| Duração da pausa e salto na abertura | [Metodologia](../experimentos/previsao_calendario_2026_11_25/README.md) e [acerto por direção](../experimentos/previsao_calendario_2026_11_25/acerto_por_direcao.csv) |
| Individuais versus pool com horizontes escolhidos em 2024 | [Comparação](../experimentos/individual_vs_pool_2026_11_25/resumo_2025.csv) |
| Individuais e pool com atributos originais ou simplificados | [Comparação com horizontes escolhidos](../experimentos/apos_pausa_features_2026_11_25/resumo_2025.csv) e [comparação com 21 pregões fixos](../experimentos/apos_pausa_features_2026_11_25/horizonte_21/resumo_2025.csv) |
| Beta de baixa e outros atributos para quedas | [Resumo](../experimentos/features_baixas_2026_11_25/resumo.csv) |
| Limiar para preservar o acerto nas altas | [Calibração](../experimentos/features_baixas_2026_11_25/calibracao_limiar/resumo.csv) |
| Peso 1,04 nas baixas | [Grade de 2024](../experimentos/features_baixas_2026_11_25/pesos_baixas/grade_2024.csv) e [resultados de 2025](../experimentos/features_baixas_2026_11_25/pesos_baixas/resumo_2025.csv) |
| Classificação logística e comparação de erros | [Resumo](../experimentos/features_baixas_2026_11_25/objetivo_direcional/resumo.csv) e [erros direcionais](../experimentos/features_baixas_2026_11_25/objetivo_direcional/erros_direcionais_2025.csv) |
| 20 combinações de atributos e objetivos | [Grade de 2024](../experimentos/busca_previsao_21_2026_11_25/grade_2024.csv) e [resultados de 2025](../experimentos/busca_previsao_21_2026_11_25/resumo_2025.csv) |
| Log-retorno versus retorno simples | [Comparação na mesma escala](../experimentos/alvo_log_vs_simples_2026_11_25/resumo.csv) |
| Soma de 21 previsões diárias com origem fixa | [Comparação com o alvo direto](../experimentos/pool_diario_acumulado_2026_11_25/resumo.csv) |
| Médias e frequências históricas | [Resultados](../experimentos/media_historica_2026_11_25/resumo.csv) |
| ATR, ADXR, MACD, EMA, SMA e RSI do artigo | [Resultados](../experimentos/fatores_artigo_2026_11_25/resumo.csv) |
| Amplitude, calibração e árvores mais flexíveis | [Resultados](../experimentos/amplitude_previsoes_2026_11_25/resumo.csv) |
| Previsão de mercado mais retorno residual | [Modelo de fatores](../experimentos/modelo_fatores_21_2026_11_25/resumo.csv) |
| Objetivo de ranking versus ranking do pool | [Resumo da ordenação](../experimentos/ranking_acoes_2026_11_25/resumo_2025.csv) |

## Quantis, visualizações e regras de decisão

| Teste ou visualização | Evidência principal |
|---|---|
| P10, P50 e P90 com cobertura por cauda | [Resumo](../experimentos/quantis_pool_21_2026_11_25/resumo_2025.csv) e [gráficos por ação](../experimentos/quantis_pool_21_2026_11_25/quantis_vs_realizado_por_ativo_2025.png) |
| P90 se alta, P10 se baixa | [Resumo da regra](../experimentos/quantis_pool_21_2026_11_25/resumo_limites_por_sinal_2025.csv) e [gráficos por ação](../experimentos/quantis_pool_21_2026_11_25/limites_por_sinal_vs_realizado_por_ativo_2025.png) |
| Proporção calibrada entre P50 e o limite | [Validação e teste](../experimentos/quantis_pool_21_2026_11_25/interpolacao/) |
| Ajuste da P50 pelo ranking do pool | [Validação e teste](../experimentos/quantis_pool_21_2026_11_25/ranking_calibrado/) |
| Previsão direta versus realizado | [Gráfico geral](../experimentos/features_baixas_2026_11_25/pesos_baixas/previsao_vs_realizado_2025.png) e [gráficos por ação](../experimentos/features_baixas_2026_11_25/pesos_baixas/previsao_vs_realizado_por_ativo_2025.png) |
