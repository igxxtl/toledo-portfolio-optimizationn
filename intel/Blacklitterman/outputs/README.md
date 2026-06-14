# Saídas do pipeline Black-Litterman

Todos os CSV e PNG gerados pelos scripts ficam em `outputs/`, organizados por etapa.

```
outputs/
├── pipeline/       # pipeline.py — Q, Ω, π, μ posterior, pesos
├── backtest/       # plot_ganho_estimado.py — séries de capital
├── score/          # score_modelo.py — métricas e score 0–100
├── markowitz/      # comparar_bl_vs_markowitz.py — BL vs Markowitz
├── figures/        # gráficos PNG
└── standalone/     # scripts Q.py, omega.py, PI_incerteza.py
    ├── q/
    ├── omega/
    └── prior/
```

Os caminhos são definidos em `config.py`.
