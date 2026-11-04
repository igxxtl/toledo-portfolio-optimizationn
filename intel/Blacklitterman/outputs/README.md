# Resultados da execução atual

Execução em 3 de novembro de 2026 do pipeline principal Black-Litterman com 20 ativos. Janela de backtest: 2025. Capital inicial: R$ 100.000; taxa fixa de referência: 15% a.a.; lag: 1 pregão; custo de turnover: 10 bps. Configuração registrada em `run_config.json`.

## Arquivos para análise

- `figures/bl_hibrido_ganho_estimado.png`: gráfico principal indicado pelo autor, regenerado nesta execução.
- `backtest/bl_hibrido_ganho_comparativo.csv`: capitais estimados, realizados, Markowitz e referência.
- `backtest/bl_hibrido_ganho_series_*.csv`: séries completas por frequência.
- `markowitz/bl_vs_markowitz_resumo.csv`: comparação com Markowitz, pesos iguais e prior.
- `markowitz/bl_vs_baselines_custo_0_vs_10bps.csv`: sensibilidade a custos.
- `score/bl_hibrido_rebalance_resumo.csv`: métricas diária, semanal e mensal.
- `score/sensibilidade_universo.csv`: full versus sem VALE3; frequência semanal conforme o padrão implementado em BLParams.
- `pipeline/`: views, incertezas, prior, posterior e pesos desta rodada.
- `run_manifest.json`: estado, módulos, versões, hashes dos dados/código e retrato das capitalizações de mercado.
- `verification.json`: conferências e capitais finais reconciliados.

## Proveniência

Foram executados, nesta ordem: pipeline.py, plot_ganho_estimado.py, score_modelo.py, comparar_bl_vs_markowitz.py (incluindo custo zero) e sensibilidade_universo.py. Dados históricos, previsões XGB e notícias classificados já existentes foram reutilizados. Não houve novo treinamento XGB, classificação de notícias, busca Optuna nem seleção top-k.

As capitalizações de mercado foram consultadas novamente e compartilhadas por todas as etapas; por isso os resultados podem diferir das figuras antigas. Os pesos consultados estão em `run_manifest.json`. O período de preços permanece 2025.

Os arquivos antigos de otimização, experimentos top-k, baselines históricos e varreduras de universo foram retirados desta pasta. O backup está fora do repositório, não deve ser misturado com esta rodada. Os snapshots numéricos de outros READMEs e de TCC_MATERIAL_REPOSITORIO.md são históricos; para esta execução, use os CSVs desta pasta.

As fórmulas do programa foram preservadas: indicadores chamados Sharpe e retorno total podem usar convenções diferentes entre módulos. Consulte backtest.py e score_modelo.py ao interpretar essas colunas. O capital final e o ganho sobre o aporte são reconciliados em verification.json.

## Sensibilidade ao número de ativos (n = 1 a 19)

A execução walk-forward compara duas formas de seleção/alocação, com rebalanceamento mensal, teste em 2025, capital inicial de R$ 100.000 e custo de transação de 10 bps:

- `score/sensibilidade_tamanho_universo_pred_wf_classical.csv`: seleção independente — BL por previsões médias recentes; Markowitz por Sharpe histórico recente.
- `score/sensibilidade_tamanho_universo_pred_wf_same_universe.csv`: ambos recebem o mesmo conjunto selecionado pelo BL, isolando a comparação entre alocações.
- `figures/sensibilidade_tamanho_universo_pred_wf_classical.svg` e `...same_universe.svg`: capital final por número de ativos.
- `sensibilidade_tamanho_universo_manifest.json`: critérios e parâmetros desta rodada.

Para reproduzir a faixa de 1 a 19 ativos: `python -m intel.Blacklitterman.sensibilidade_tamanho_universo --walk-forward --max-assets 19`.
