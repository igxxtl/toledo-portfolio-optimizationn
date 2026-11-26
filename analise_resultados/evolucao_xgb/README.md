# Auditoria, features e metodologia XGBoost

Os arquivos atuais de dados, modelos, previsões e resultados financeiros foram preservados. Esta rodada estende os testes anteriores e não altera automaticamente o TCC ou o pipeline BL.

## O que foi executado

1. Auditoria de preços/calendário e de redundância das 103 features.
2. Comparação dos conjuntos completo, enxuto (24 features), enxuto com novas informações (aproximadamente 48) e contexto ampliado de mercado (aproximadamente 60).
3. Comparação de treino expansivo, janelas móveis de dois/quatro anos, normalização do alvo pela volatilidade conhecida, modelo conjunto dos vinte ativos e modelo conjunto normalizado.
4. Comparações pareadas com o modelo original, com as candidatas dos testes anteriores e com retorno zero/média histórica.
5. Receita congelada e tentativa de obter preços posteriores ao acervo para confirmação. A consulta falhou com timeout de resolução de rede; não foram recebidas cotações utilizáveis.

Todos os alvos continuam sendo retornos numéricos. Os alvos normalizados são reconstruídos na escala original antes de avaliar erros. Features usam somente preços/volumes/contexto disponíveis no fechamento de origem. Filtros de redundância/ausência/constância são ajustados no fit purgado, sem o alvo e sem o teste.

## Dois resultados diferentes

`resumo_final.csv` e `comparacao_final_por_ativo.csv` apresentam a **receita selecionada pela validação de 2024**, sem trocar escolhas pelo desempenho de 2025.

`modelo_conjunto_exploratorio.csv` apresenta o **modelo conjunto fixo**, uma das metodologias predefinidas que mostrou resultado mais promissor em 2025. Seu destaque é posterior ao teste. Ele não substitui retroativamente a seleção de 2024 e precisa de confirmação externa. As duas receitas foram guardadas separadamente para não confundir exploração e validação.

## Limitações

2025 já foi inspecionado diversas vezes. O período até março de 2026 também já entrou na análise OOF completa. Portanto, nenhum deles é agora um teste inteiramente independente. A nova janela proposta é 09/04/2026 a 08/11/2026, mas a consulta ao Yahoo Finance via yfinance não forneceu dados por falha de resolução de rede. `disponibilidade_confirmacao.json` registra o resultado. A confirmação e a avaliação financeira da nova candidata estão pendentes; não se presume que uma melhora de RMSE aumente o retorno da carteira.

A auditoria encontrou pares duplicados entre features, uma linha de OHLC zerado por ação em 08/04/2026 (fora dos testes) e uma data de ações sem registro correspondente no IBOV em 25/01/2018. A divergência de calendário foi documentada sem reescrever o histórico nesta comparação. Volume zero no arquivo de câmbio não é tratado como falta de negociação de uma ação. Consistência de preços ajustados não comprova disponibilidade point-in-time ou ausência de revisões.

## Arquivos e reprodução

- `RELATORIO.txt`: interpretação completa e limites.
- `previsoes_finais_2025.csv`: cada previsão da receita selecionada em 2024, lado a lado com o resultado e referências.
- `efeitos_metodologia.csv`: comparação de cada método com o expansivo dentro do mesmo conjunto de features.
- `receita_congelada.json`: hashes da receita principal, anteriores à consulta de novos dados.
- `receita_pool_para_confirmacao.json`: candidata exploratória conjunta, guardada para avaliação futura e sem promoção ao pipeline.
- `../features_xgb/`: auditorias, filtros usados em cada treino, conjuntos fixos e escolhas individuais em 2024.
- `../metodologia_xgb/`: modelos fixos, seleção de features comum por horizonte para a comparação de pooling, escolhas de métodos e auditoria temporal.

```powershell
python -m unittest intel.model_training.test_experimento_melhoria intel.model_training.test_features_experimento intel.model_training.test_metodologia_xgb
python -m intel.model_training.experimento_features --audit-only
python -m intel.model_training.experimento_features
python -m intel.model_training.experimento_metodologia
python -m analise_resultados.resumir_features_metodologia
```

Dependências e hashes estão nos protocolos. São necessárias as mesmas bibliotecas da rodada anterior (XGBoost, scikit-learn, numpy, pandas, optuna importado pelo treinador reutilizado e matplotlib para figuras). Yfinance é usado somente na consulta opcional de confirmação, com cache em `tmp/`, sem alterar `dados_diarios`.
