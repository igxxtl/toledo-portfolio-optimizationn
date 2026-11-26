# Testes isolados de melhoria do XGBoost

O pipeline atual e os artefatos de `criacao_modelo_xgb` não são substituídos. Os hashes dos arquivos históricos de previsões e métricas são verificados ao terminar.

## Protocolo

Vinte ativos, horizontes de 5/10/21 pregões, mesmas features do treinador existente e ranks cross-sectionais do mesmo universo. São oito configurações predefinidas:

| Família | Perfil | Peso temporal lambda |
|---|---|---|
| Erro quadrático | regular: profundidade 4, min_child_weight 10, alpha 0,1, lambda de regularização 1 | 0 / 1 / 5 |
| Erro quadrático | conservador: profundidade 2, min_child_weight 30, alpha 0,5, lambda de regularização 10 | 0 / 1 / 5 |
| Erro absoluto | conservador | 0 / 1 |

Lambda temporal zero significa pesos iguais, um significa decaimento moderado e cinco reproduz a intensidade forte da configuração anterior. Não confundir lambda temporal com `reg_lambda`. O perfil conservador muda um conjunto de parâmetros, não um parâmetro isolado.

As configurações são escolhidas **por RMSE de 2024, dentro de cada ativo e horizonte**, usando dois folds semestrais expansivos. Não se escolhe horizonte por RMSE bruto, pois prazos diferentes representam alvos de escalas distintas. A comparação de 2025 usa quatro folds trimestrais com a configuração escolhida em 2024 fixa. O treino de cada trimestre pode incorporar apenas alvos já conhecidos de trimestres anteriores. Todas as oito configurações fixas são apresentadas para transparência, mas não se escolhe outra vencedora por seus resultados em 2025.

A purga usa a data real de maturidade do alvo, tanto na fronteira treino/teste como na fronteira fit/early stopping. O número de árvores é definido com validação interna anterior à decisão e depois há refit com todo o treino conhecido. O limite é 400 árvores; taxa 0,03; hist/max_bin=64; seed 42. A normalização de y é ajustada somente no treino de cada etapa. Não há novos downloads de dados.

## Reprodução

Dependências observadas na execução: XGBoost 3.4.1, scikit-learn 1.9.1, numpy 2.3.5, pandas 3.0.1, optuna 5.0.0 e matplotlib (somente gráficos). Optuna é importado pelo treinador reutilizado; esta rodada usa uma grade fixa, sem nova busca Optuna.

```powershell
python -m unittest intel.model_training.test_experimento_melhoria
python -m intel.model_training.experimento_melhoria
python -m analise_resultados.resumir_melhoria_xgb
```

`--resume` reutiliza casos completos apenas se o protocolo, código e hashes de entrada coincidirem. `--tickers`, `--horizons` e `--output` permitem um piloto separado. Os defaults de execução são quatro casos independentes simultâneos, com duas threads por modelo; o paralelismo não muda os critérios de seleção.

## Resultados

- `RELATORIO.txt`: resumo interpretativo e limitações após conclusão da rodada.
- `comparacao_pareada_2025.csv`: atual versus candidata escolhida em 2024 para as 60 combinações, nos mesmos alvos e datas.
- `comparacao_horizontes_atuais.csv`: mantém o horizonte usado no experimento financeiro para cada um dos vinte ativos.
- `resumo_horizontes.csv`: médias simples entre ativos, separadas por horizonte.
- `configuracoes_fixas_teste.csv`: desempenho de todas as configurações predefinidas, além dos benchmarks.
- `escolhas_validacao_2024.csv`: rastreabilidade da seleção anterior ao teste.
- `auditoria_temporal.csv`: fronteiras de maturidade, treino, early stopping e número de árvores de cada ajuste.
- `previsoes.csv` e `cases/`: todas as previsões e resultados, com checkpoints por combinação.
- `efeitos_controlados.csv`: contrastes de pesos/perfis/perdas dentro do mesmo protocolo novo.
- `protocolo.json`: parâmetros, versões, calendário e hashes congelados antes da execução.

## Interpretação

A comparação principal mede qualidade preditiva, sem atualizar o BL ou o backtest. Reduzir erro contra o modelo antigo não basta: comparar também com retorno zero e média histórica observável. MAE e RMSE estão em log-retorno; multiplicar por cem não é uma conversão exata para retorno simples.

O baseline original inclui escolhas históricas entre regressor e classificador; as candidatas usam regressão e novo calendário de treino. A diferença contra o atual não isola uma única mudança. A comparação entre configurações dentro do novo protocolo ajuda a distinguir os efeitos das escolhas.

2025 já foi inspecionado na pesquisa. A escolha numérica das configurações usa somente a validação anterior, mas o desenho da rodada foi motivado pelos resultados já observados. Portanto, esses resultados são retrospectivos. Não são prova independente de generalização nem testes de significância com observações independentes. Não promover automaticamente uma candidata para o pipeline por vencer em alguns ativos do teste.
