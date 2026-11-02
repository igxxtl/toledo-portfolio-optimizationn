# Black-Litterman (`intel/Blacklitterman`)

Pipeline de otimização de portfólio com views **XGBoost** + sentimento. Universo vigente: **20 ações** da B3. Parâmetros oficiais em [`config.py`](config.py), alinhados a `outputs/optimize/melhorias_bl_wf_best.json` (walk-forward, **k\* = 5**, rebalance **monthly**).

Guia de leitura: [`COMO_FUNCIONA.md`](COMO_FUNCIONA.md). Mapa de saídas: [`outputs/README.md`](outputs/README.md).

## Estrutura

```
intel/Blacklitterman/
├── config.py                       # ★ Parâmetros e caminhos
├── params.py                       # BLParams (hiperparâmetros tunáveis)
├── pipeline.py                     # ★ Q → Ω → π → μ → pesos
├── xgb_views.py                    # Views Q (XGB + gate de notícias)
├── sentiment.py                    # Agregação diária do JSON GPT
├── omega_calc.py                   # Ω dinâmico + hard prior
├── prior.py                        # π, posterior BL, rebalance
├── weights.py                      # Projeção long-only com teto
├── backtest.py                     # Capital, turnover, métricas
├── plot_ganho_estimado.py
├── score_modelo.py
├── comparar_bl_vs_markowitz.py
├── otimizar_parametros.py          # Optuna (val 2024 / teste 2025)
├── aplicar_melhores_params.py
├── melhorias_bl_wf.py              # ★ k* + regularização (params vigentes)
├── sensibilidade_universo.py
├── sensibilidade_tamanho_universo.py
├── Q.py / omega.py / PI.py / PI_incerteza.py
└── outputs/
```

## Fluxo

```mermaid
flowchart LR
    XGB["criacao_modelo_xgb/*_predicoes.csv"] --> QV["xgb_views.py"]
    NEWS["noticias sentimento"] --> SENT["sentiment.py"]
    QV --> Q["Vetor Q"]
    SENT --> Q
    Q --> OMEGA["omega_calc.py"]
    DD["dados_diarios/"] --> PI["prior.py"]
    OMEGA --> BL["Posterior BL"]
    PI --> BL
```

## Como executar

```bash
cd intel/Blacklitterman

python pipeline.py
python plot_ganho_estimado.py
python score_modelo.py
python comparar_bl_vs_markowitz.py

# Pesquisa
python otimizar_parametros.py
python melhorias_bl_wf.py
python sensibilidade_tamanho_universo.py
```

## View Q

Cobertura \(c = \min(1,\ n / \texttt{NEWS\_COUNT\_MIN})\), com `NEWS_COUNT_MIN = 5`:

```
Q_h = [α + (1-α)·(1-c)] · ret_pred + (1-α)·c · sentiment_component
Q   = Q_h / h
```

| Coluna | Descrição |
|--------|-----------|
| `ret_pred` / `ret_real` | Log-retorno XGB no horizonte \(h\) |
| `Q` | View já em escala diária |
| `news_count` / `news_coverage` | Manchetes do dia e cobertura do gate |
| `q_source` | Sempre `xgb` |

α vigente = **0,386251**. Sem cobertura, Q é puro XGB. Com `HARD_PRIOR_WHEN_LOW_NEWS = True`, \(n < 5\) zera a confiança da view (dia fica no prior).

## Parâmetros vigentes

| Parâmetro | Valor |
|-----------|-------|
| Universo | 20 tickers `.SA` |
| `PERIOD_START` / `END` | 202501 / 202512 |
| `TAU` | 0,0147235 |
| `ROLLING_MIN_OBS` | 80 |
| `MAX_WEIGHT_PER_ASSET` | 0,341677 |
| `W_MODEL_CONF` / `W_NEWS_CONF` | 0,659122 / 0,340878 |
| `NEWS_OMEGA_BOOST` | 3,79598 |
| Rebalance do tuning | monthly |
| Custo / lag | 10 bps / 1 dia |
| Capital / Selic | R$ 100.000 / 15% a.a. |

## Convenções

- Config centralizado em `config.py`
- Imports `try: from .module` / `except ImportError` para pacote e script
- Pesos long-only em `weights.py`; CSV em `io_utils.py`

## Comparação de limiares de notícias (n = 0 a 5)

A partir da raiz do repositório:

```bash
python -m intel.Blacklitterman.simular_cenarios_noticias
python -m unittest intel.Blacklitterman.test_metricas_cobertura_noticias
```

Gera duas tabelas em `outputs/score/`: `cenarios_noticias_percentuais.csv`
(cobertura) e `cenarios_noticias_scores.csv` (scores e backtest).
O comando **não sobrescreve** os CSVs do pipeline padrão; `NEWS_COUNT_MIN=5`
continua vigente em `config.py`. Os scores são calculados para rebalanceamento
`monthly`, salvo se outro `modo` for passado a `simular_cenarios`.

**Interpretação da cobertura:**

- `pct_noticias_dentro_padrao`: soma de `news_count` em pares ticker-dia com
  `news_count >= n`, dividida pela soma total de `news_count`.
  Não equivale à porcentagem de manchetes **únicas** do JSON: manchetes
  compartilhadas pela mesma empresa podem aparecer em mais de um ticker.
  Sem notícias no período, o percentual é indefinido (NaN).
- `pct_ativo_dias_ge_n` e `pct_ativo_dias_gt_n`: proporção de pares
  ticker-dia com pelo menos `n` e com **mais de `n`** notícias, respectivamente.
  O denominador inclui os pares sem notícias.
- `pct_dias_com_algum_ativo_ge_n` e `pct_dias_com_algum_ativo_gt_n`:
  dias em que **algum ativo** atinge ou excede o limiar.
- `pct_dias_com_total_noticias_ge_n` e `pct_dias_com_total_noticias_gt_n`:
  dias em que a **soma** das notícias por ativo atinge ou supera o limiar.
  Essa soma pode contar a mesma manchete em múltiplos ativos.
- `pct_dias_com_noticias`, `media_noticias_por_dia` e
  `media_noticias_por_ativo_dia` quantificam frequência e volume.
- `pct_elegiveis` e `pct_abaixo_limiar` são aliases históricos das
  estatísticas inclusivas de pares ticker-dia, mantidos por compatibilidade.

Os dias do denominador são somente as datas presentes em Q; não incluem
fins de semana nem datas sem qualquer registro de view. Para `n=0`,
todos os pares são elegíveis pelo teste `>= 0`, mesmo sem manchetes.
O critério `> 0` continua distinguindo pares que tiveram notícia.
