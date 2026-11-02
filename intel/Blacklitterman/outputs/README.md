# Saídas do pipeline Black-Litterman

CSV e PNG ficam em `outputs/`. Caminhos em [`../config.py`](../config.py).

Parâmetros oficiais do código: `config.py` = `optimize/melhorias_bl_wf_best.json` (k\* = 5).

`optimize/optimization_summary.json` e `markowitz/bl_vs_markowitz_resumo.csv` são de um Optuna anterior (teto ≈ 42,7%, gate de 2 notícias). Não misturar com o resultado do k\*.

```
outputs/
├── pipeline/       # pipeline.py — Q, Ω, π, Σ, μ, pesos
├── backtest/       # plot_ganho_estimado.py — séries de capital
├── score/          # score, sensibilidade de universo/tamanho, k*
├── markowitz/      # BL vs Markowitz / equal-weight / prior / custos
├── optimize/       # Optuna e best params do walk-forward
├── figures/        # PNG
├── standalone/     # Q.py, omega.py, PI_incerteza.py
│   ├── q/
│   ├── omega/
│   └── prior/
└── baseline_*.json # snapshot histórico (pré-mudanças)
```

## Arquivos a citar na versão atual

| Arquivo | O que é |
|---------|---------|
| `optimize/melhorias_bl_wf_best.json` | Params vigentes + k\* = 5 |
| `score/melhorias_bl_wf_kstar_metricas.csv` | OOS 2025: BL top-5 vs híbrido vs Markowitz |
| `score/melhorias_bl_wf_oos_sweep.csv` | Sweep k = 1…20 (não usar o melhor k ex-post como resultado principal) |
| `figures/melhorias_bl_wf_kstar_capital.png` | Capital do experimento k\* |
| `pipeline/bl_hibrido_q_long.csv` | Views diárias (cobertura de notícias) |
| `score/sensibilidade_tamanho_universo.csv` | Universo crescente; gerado com o Optuna anterior |

## Resultado do k\* (teste 2025, params vigentes)

| Estratégia | Sharpe | Retorno | Capital | Drawdown |
|------------|--------|---------|---------|----------|
| BL no top-5 | 0,85 | +15,2% | R$ 114,0 mil | −9,6% |
| Híbrido rank + Markowitz | 0,40 | +7,4% | R$ 106,0 mil | −11,3% |
| Markowitz clássico no top-5 | −0,12 | −2,5% | R$ 96,2 mil | −18,4% |

Capital inicial R$ 100.000, 249 dias, custo do experimento conforme o script de melhorias.
