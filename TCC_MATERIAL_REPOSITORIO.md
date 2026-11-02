# Material técnico do repositório — apoio ao TCC

**Projeto:** Toledo Portfolio Optimization  
**Repositório:** `toledo-portfolio-optimizationn`  
**Objetivo deste documento:** descrever, de forma extensiva, a finalidade, a arquitetura, os pipelines, as premissas, as fórmulas, os parâmetros e as saídas do código, para uso em sumário, metodologia e apêndices do TCC.

> **Fonte de verdade:** o código em `intel/*/config.py` e os módulos Python.  
> **Versão vigente (set/2026):** 20 tickers; params de `melhorias_bl_wf_best.json` (k\* = 5, rebalance monthly). α ≈ 0,386, τ ≈ 0,0147, teto ≈ 34,2%, `NEWS_COUNT_MIN = 5`, hard prior ligado. READMEs alinhados a esta config. `optimization_summary.json` é um Optuna anterior — não misturar.

> **Atualização experimental — 02/11/2026:** o arquivo `Blacklitterman.zip` fornecido para esta atualização contém uma **reexecução local** com `NEWS_COUNT_MIN = 1` e uma varredura exploratória de `n = 0, 1, 2, 3, 4, 5` no ano de 2025. Isso **não substitui automaticamente** o parâmetro histórico `n = 5` gravado em `melhorias_bl_wf_best.json`, cuja seleção ocorreu na validação de 2024. As duas experiências e seus resultados são descritos separadamente nas seções 11.4, 13 e 14.5.

---

## Índice

1. [Finalidade e pergunta de pesquisa](#1-finalidade-e-pergunta-de-pesquisa)
2. [Enquadramento teórico resumido](#2-enquadramento-teórico-resumido)
3. [Arquitetura do repositório](#3-arquitetura-do-repositório)
4. [Universo de ativos e dados](#4-universo-de-ativos-e-dados)
5. [Visão geral do pipeline end-to-end](#5-visão-geral-do-pipeline-end-to-end)
6. [Pipeline A — dados e treino XGBoost](#6-pipeline-a--dados-e-treino-xgboost)
7. [Pipeline B — notícias e sentimento](#7-pipeline-b--notícias-e-sentimento)
8. [Pipeline C — Black-Litterman](#8-pipeline-c--black-litterman)
9. [Avaliação, score e baselines](#9-avaliação-score-e-baselines)
10. [Otimização de hiperparâmetros (Optuna)](#10-otimização-de-hiperparâmetros-optuna)
11. [Análises de sensibilidade](#11-análises-de-sensibilidade)
12. [Premissas (lista explícita)](#12-premissas-lista-explícita)
13. [Parâmetros oficiais (config atual)](#13-parâmetros-oficiais-config-atual)
14. [Resultados materializados (snapshots)](#14-resultados-materializados-snapshots)
15. [Ordem de execução e dependências](#15-ordem-de-execução-e-dependências)
16. [Mapa de saídas](#16-mapa-de-saídas)
17. [Limitações e inconsistências documentais](#17-limitações-e-inconsistências-documentais)
18. [Sugestão de estrutura de capítulos do TCC](#18-sugestão-de-estrutura-de-capítulos-do-tcc)
19. [Glossário](#19-glossário)
20. [Referências internas do repositório](#20-referências-internas-do-repositório)

---

## 1. Finalidade e pergunta de pesquisa

O repositório implementa um **pipeline de pesquisa** (não uma aplicação web) para **otimização de portfólio de ações da B3**, combinando:

| Fonte | Tecnologia | Papel |
|-------|------------|--------|
| Views quantitativas | XGBoost (walk-forward OOF) | Log-retorno esperado por ativo |
| Views qualitativas | GPT sobre manchetes | Viés positivo / negativo / neutro |
| Equilíbrio de mercado | Prior Black-Litterman | Opinião implícita via market cap + Σ |
| Alocação | Posterior BL + restrições | Pesos long-only com teto por ativo |
| Avaliação | Backtest + score + baselines | Capital, Sharpe, drawdown vs Markowitz/Selic |

### Pergunta operacional

> Como alocar capital entre um conjunto de ações líquidas da B3, combinando previsões de machine learning, sentimento de notícias e a teoria de equilíbrio de mercado, de forma que a **confiança** em cada opinião controle o quanto ela altera o portfólio?

### Decisão arquitetural explícita

O único modelo quantitativo que alimenta o vetor de views \(Q\) é o **XGBoost**. O módulo Black-Litterman **não treina** o XGB: consome artefatos já gerados em `criacao_modelo_xgb/*_predicoes.csv`.

### Frase-síntese (abstract em uma linha)

O projeto combina previsões out-of-fold de XGBoost e sentimento de notícias (GPT) em views Black-Litterman com incerteza dinâmica, gate de 5 manchetes (hard prior), prior de equilíbrio rolling e restrições long-only, para alocar ações da B3. A config vigente opera o BL no top-k (k\* = 5, escolhido em 2024) e reporta o teste 2025 contra Markowitz e Selic.

---

## 2. Enquadramento teórico resumido

### 2.1 Por que não só Markowitz?

O mean-variance clássico (Markowitz) estima retornos esperados pela média histórica. Médias de retorno são ruidosas; a otimização responde com pesos extremos e instáveis. O modelo **Black-Litterman (1992)** parte do equilíbrio de mercado e permite incorporar **views** do investidor com níveis de confiança.

### 2.2 Por que views de ML + sentimento?

- **ML (XGBoost):** captura padrões não lineares em features de preço, volume, relativo ao Ibovespa e macros setoriais; produz uma opinião quantitativa calibrada em escala de log-retorno.
- **Sentimento (GPT):** injeta informação qualitativa de manchetes que o preço pode ainda não ter absorvido plenamente; sozinho é ruidoso e precisa ser escalado para unidade de retorno.
- **BL:** mistura as duas fontes com o prior de mercado, ponderando pela matriz de incerteza \(\Omega\).

### 2.3 Hipótese operacional implícita no código

Um portfólio que combina (i) views OOF de XGBoost, (ii) sentimento diário de manchetes e (iii) prior de equilíbrio, com \(\Omega\) dinâmico e restrições long-only, pode gerar alocações avaliáveis contra Markowitz puro e Selic em janela out-of-sample (**2025**), após tuning em validação (**2024**).

---

## 3. Arquitetura do repositório

```
toledo-portfolio-optimizationn/
├── .env                              # OPENAI_API_KEY (gitignored)
├── .gitignore
├── requirements.txt
├── README.md                         # visão geral (parcialmente desatualizada)
├── TCC_MATERIAL_REPOSITORIO.md       # este documento
│
├── dados_diarios/                    # OHLCV diário + features vs IBOV
│   ├── ABEV3.csv … WEGE3.csv         # 20 ações
│   ├── IBOV.csv                      # benchmark
│   └── USDBRL.csv                    # macro cambial
│
├── tickers_data/                     # barras horárias B3 60 min
│   └── BMFBOVESPA_DLY_<TICKER>, 60.csv
│
├── criacao_modelo_xgb/               # artefatos XGB (entrada Q do BL)
│   ├── <TICKER>_predicoes.csv        # ★ walk-forward OOF
│   ├── <TICKER>_h{5,10,21}d_*        # artefatos por horizonte
│   ├── metricas_vencedores.csv       # horizonte/modelo eleito
│   ├── q_vetor.csv / omega_base.csv
│   └── *.pkl, *.json, *.png, shap…
│
└── intel/                            # todo o código de pesquisa
    ├── __init__.py
    ├── model_training/               # download → features → treino XGB
    ├── data_acquisition/             # RSS → limpeza → sentimento GPT
    └── Blacklitterman/               # Q → Ω → π → μ → pesos → avaliação

**Extensão experimental:** `intel/Blacklitterman/simular_cenarios_noticias.py` executa o sweep de `n=0..5` (disponível na branch de simulação, não necessariamente na `main`).
```

### Separação de responsabilidades

| Camada | Pasta | Produz | Consome |
|--------|-------|--------|---------|
| Dados de preço | `dados_diarios/`, `tickers_data/` | Séries OHLCV | Treino + prior + backtest |
| Inteligência quantitativa | `intel/model_training/` → `criacao_modelo_xgb/` | Predições OOF | BL (`xgb_views.py`) |
| Inteligência qualitativa | `intel/data_acquisition/` | JSON com sentimento | BL (`sentiment.py`) |
| Otimização e avaliação | `intel/Blacklitterman/` | Pesos, μ, métricas, figuras | Relatório / TCC |

Não há notebooks (`.ipynb`) no fluxo principal. Tudo é script CLI Python. Saídas ficam em `intel/Blacklitterman/outputs/`.

### Módulos Python por pasta

#### `intel/model_training/`

| Arquivo | Papel |
|---------|--------|
| `config.py` | Paths, janelas de features IBOV, datas de download |
| `download_daily.py` | yfinance → `dados_diarios/` |
| `prepare_features.py` | Features relativas ao IBOV |
| `train_xgb.py` | Walk-forward OOF, Optuna, horizontes, clf vs reg |
| `pipeline.py` | Orquestra download → features → train |

#### `intel/data_acquisition/`

| Arquivo | Papel |
|---------|--------|
| `config.py` | Paths JSON, 20 tickers RSS, modelo GPT, filtros de título |
| `cata_link_rss.py` | Google News RSS → `noticias_b3.json` |
| `limpa_duplicados.py` | Dedup por título |
| `limpa_noticias.py` | Remove títulos sem menção ao ticker/empresa |
| `analise_sentimento.py` | Classificação positivo/negativo/neutro via OpenAI |
| `traduz_link_do_rss.py` | Resolve `link_google` → link real (Playwright) |
| `extrair_search_xml.py` | Extrai `search.xml` → JSON/CSV |
| `ordenar_noticias_por_data.py` | Ordena extract por data |

#### `intel/Blacklitterman/`

| Arquivo | Papel |
|---------|--------|
| `config.py` | Paths, universo, α/τ/Ω, Optuna, custos, saídas |
| `params.py` | Dataclass `BLParams` (hiperparâmetros tunáveis) |
| `pipeline.py` | ★ Entry principal: Q → Ω → π → μ → pesos |
| `xgb_views.py` | Carrega predições, combina com sentimento → Q |
| `sentiment.py` | Agrega sentimento diário do JSON GPT |
| `omega_calc.py` | Ω dinâmico |
| `prior.py` | π, posterior BL, controles, rebalance |
| `weights.py` | Projeção long-only com teto |
| `io_utils.py` | CSV com coluna de data |
| `backtest.py` | Séries de capital, turnover/custo, métricas |
| `plot_ganho_estimado.py` | Backtest estimado vs realizado vs Selic |
| `score_modelo.py` | Score composto 0–100 |
| `comparar_bl_vs_markowitz.py` | BL vs Markowitz / equal-weight / prior-only |
| `otimizar_parametros.py` | Optuna (val 2024; teste 2025) |
| `aplicar_melhores_params.py` | Grava best params e reavalia |
| `sensibilidade_universo.py` | Full vs sem ativos fracos |
| `Q.py` / `omega.py` / `PI.py` / `PI_incerteza.py` | Standalone (debug / VWAP) |
| `_check_news_gate.py` | Self-check do gate de notícias |

---

## 4. Universo de ativos e dados

### 4.1 Universo atual (20 ações)

Definido em `intel/Blacklitterman/config.py` e espelhado em `intel/data_acquisition/config.py`:

| Ticker | Observação típica no pipeline |
|--------|-------------------------------|
| ABEV3 | Consumo / bebidas |
| B3SA3 | Bolsa / infraestrutura financeira |
| BBAS3 | Banco público |
| BBDC4 | Banco privado (PN) |
| CMIG4 | Energia (Cemig) |
| EQTL3 | Energia (Equatorial) |
| GGBR4 | Siderurgia (Gerdau) |
| ITUB4 | Banco privado (PN) |
| LREN3 | Varejo (Renner) |
| MGLU3 | Varejo (Magalu) — alta volatilidade |
| PETR3 / PETR4 | Petróleo (ON/PN) |
| PRIO3 | Petróleo (independente) |
| RADL3 | Farmácia (Raia Drogasil) |
| RAIL3 | Logística (Rumo) |
| SANB11 | Banco (unit Santander) |
| SBSP3 | Saneamento (Sabesp) |
| SUZB3 | Celulose (Suzano) |
| VALE3 | Mineração — alvo de sensibilidade (hit direcional fraco) |
| WEGE3 | Bens de capital (WEG) |

No prior/yfinance os tickers usam sufixo `.SA` (ex.: `VALE3.SA`).

### 4.2 Dados auxiliares

| Arquivo | Uso |
|---------|-----|
| `dados_diarios/IBOV.csv` | Features relativas ao mercado; contexto no treino |
| `dados_diarios/USDBRL.csv` | Macro cambial (treino XGB) |
| Macros setoriais opcionais | `brent`, `iron_ore`, `cellulose`, `sugar` (se existirem) |

### 4.3 `tickers_data/` (horário)

CSVs `BMFBOVESPA_DLY_<TICKER>, 60.csv`. Servem para:

1. Definir o universo do download diário.
2. Alimentar `PI_incerteza.py` (prior alternativo com retornos VWAP intradía) — **não** é o caminho principal do `pipeline.py`.

### 4.4 Sensibilidade de universo

```python
ATIVOS_SENSITIVITY_DROP = ["VALE3.SA"]
```

Script: `sensibilidade_universo.py` — compara portfólio full vs sem ativos com acurácia direcional fraca no recorte OOF.

### 4.5 Janelas temporais

| Uso | Início | Fim |
|-----|--------|-----|
| Download diário (treino) | 2018-01-01 | hoje |
| Coleta de notícias | 2023-01-01 | 2025-12-31 |
| Histórico do prior Σ/π | 2023-01-01 | 2025-12-31 |
| Views Q / backtest de comparação | 2025-01 | 2025-12 |
| Optuna validação | 2024-01 | 2024-12 |
| Optuna teste (congelado) | 2025-01 | 2025-12 |

---

## 5. Visão geral do pipeline end-to-end

```text
[1] Preços (yfinance / horários)
        ↓
[2] Features vs IBOV (+ técnicos / macro / ranks)
        ↓
[3] Treino XGB walk-forward OOF (horizontes 5/10/21; reg vs clf)
        ↓
[4] Notícias Google News RSS → limpeza → GPT (pos/neg/neu)
        ↓
[5] Views Q híbridas (XGB + sentimento + gate de cobertura)
        ↓
[6] Ω dinâmico (erro do modelo + notícias + τ·σ)
        ↓
[7] Prior π = δ Σ w_mkt (rolling, ≥80 obs)
        ↓
[8] Posterior BL μ; w ∝ Σ⁻¹μ; long-only + teto
        ↓
[9] Rebalance daily / weekly / monthly
        ↓
[10] Backtest (lag 1 dia, custo 10 bps) + score + vs Markowitz + Optuna
```

Diagrama lógico do BL:

```mermaid
flowchart TB
    subgraph Entradas
        XGB["Predições XGBoost<br/>*_predicoes.csv"]
        NEWS["Notícias + sentimento<br/>JSON GPT"]
        PRICES["Preços diários<br/>dados_diarios/"]
    end

    subgraph Pipeline["pipeline.py"]
        Q["1. Vetor Q"]
        OMEGA["2. Omega Ω"]
        PI["3. Prior π"]
        POST["4. Posterior μ"]
        W["5. Pesos + restrições"]
    end

    subgraph Avaliação
        PLOT["plot_ganho_estimado.py"]
        SCORE["score_modelo.py"]
        MKZ["comparar_bl_vs_markowitz.py"]
    end

    XGB --> Q
    NEWS --> Q
    Q --> OMEGA
    PRICES --> PI
    Q --> POST
    OMEGA --> POST
    PI --> POST
    POST --> W
    W --> PLOT
    W --> SCORE
    W --> MKZ
```

---

## 6. Pipeline A — dados e treino XGBoost

**Pasta:** `intel/model_training/`  
**Orquestração:** `python pipeline.py` (ou etapas `download` / `features` / `train`)  
**Saída:** `criacao_modelo_xgb/`

### 6.1 Download (`download_daily.py`)

- Fonte: **yfinance**
- Início: `DAILY_START_DATE = "2018-01-01"`; fim = hoje (`None`)
- Universo: stems presentes em `tickers_data/`
- Colunas: `date, open, high, low, close, volume, ticker`
- Destino: `dados_diarios/<TICKER>.csv`

### 6.2 Features relativas ao Ibovespa (`prepare_features.py`)

Janelas: **`FEATURE_WINDOWS = [3, 5, 14, 21, 42, 63]`**.

Para cada janela \(N\):

| Feature | Definição (ideia) |
|---------|-------------------|
| `spread_retorno_Nd` | Log-retorno \(N\)d do ativo − log-retorno \(N\)d do IBOV |
| `dist_sma_Nd` | Distância à SMA do ativo relativa à do mercado |
| `spread_vol_Nd` | Volatilidade do ativo − vol do mercado |
| `beta_Nd` | \(\beta\) móvel: \(\mathrm{Cov}(R_a,R_m)/\mathrm{Var}(R_m)\) |
| `spread_volratio_Nd` | Volume relativo vs mercado |
| `z_alpha_Nd` | Z-score do excesso de retorno vs IBOV |
| `trend_mercado_Nd` | Tendência (log-retorno) do IBOV em \(N\) dias |
| `risco_mercado_Nd` | Volatilidade do IBOV |
| `z_mercado_Nd` | Z-score de preço do IBOV |

Também grava `log_ret` e `ibov_log_ret`.

### 6.3 Treino (`train_xgb.py`) — núcleo do vetor Q

**Objetivo declarado no código:** prever o log-retorno de cada ativo no período \(t+h\) e usar o output como vetor \(Q\) (opiniões) do Black-Litterman.

#### Torneio por ticker

1. Horizontes: **`HORIZONS = [5, 10, 21]`** (≈ 1 semana / 2 semanas / 1 mês).
2. Critério de eleição: **`dir_acc_pct`** (acurácia direcional OOF).
3. Compete **regressor** (`XGBRegressor`) vs **classificador 3 classes** (`XGBClassifier`) quando `USE_CLASSIFICATION=True`.
4. O vencedor grava artefatos canônicos (`TICKER_predicoes.csv`, model, params, etc.).

#### Classificador → retorno esperado

\[
Q_{\mathrm{clf}} = P(\mathrm{up})\,\mu_{\mathrm{up}}
  + P(\mathrm{neutral})\,\mu_{\mathrm{neutral}}
  + P(\mathrm{down})\,\mu_{\mathrm{down}}
\]

#### Walk-forward e regularização temporal

| Parâmetro | Valor | Significado |
|-----------|-------|-------------|
| `WF_TRAIN_INICIAL` | 0.60 | Fração inicial de treino |
| `WF_N_FOLDS` | 5 | Folds walk-forward |
| `WF_INTERNAL_SPLIT` | 0.80 | Split interno para early stopping |
| `WF_EMBARGO_FATOR` | 1.0 | Embargo proporcional ao horizonte |
| `SEED` | 42 | Reprodutibilidade |
| `NORMALIZAR_TARGET` | True | `StandardScaler` no target |
| `USE_SAMPLE_WEIGHTS` | True | Decaimento temporal |
| `SAMPLE_WEIGHT_LAMBDA` | 5.0 | \(w_t = e^{\lambda t/n}\) (último ≫ primeiro) |

#### Optuna no treino (por ticker/horizonte)

- Trials: **40**; CV temporal: **4** folds.
- Espaço aproximado: `learning_rate ∈ [0.01, 0.20]`, `max_depth ∈ [3, 7]`, `subsample`, `colsample_bytree`, `min_child_weight`, `reg_alpha`, `reg_lambda`; `n_estimators` até 800 no search (default base 1000).

#### Famílias de features (além do IBOV-spread)

1. **Autoregressivas:** lags de retorno `[1, 5, 21]`; momentum `[21, 63, 126, 252]`.
2. **Risco/volatilidade:** vol `[21, 63]`; ATR 21.
3. **Volume / VWAP.**
4. **Técnicos:** RSI, EMA, Bollinger (\(k=2\)).
5. **Contexto de mercado:** beta, z-alpha, risco/trend do mercado.
6. **Macro setorial (opcional):** PETR/PRIO→brent; VALE/GGBR→iron_ore; SUZB→cellulose; etc.
7. **Rank cross-seccional:** percentil de retorno no universo `[21, 63, 126]`.

Flags opcionais (desligadas por padrão): `USE_QUANTILE`, `USE_LGB_ENSEMBLE`. SHAP ligado (`USE_SHAP=True`) quando a lib está instalada.

#### Penalidade de incerteza por baixa acurácia direcional

Se `dir_acc < 52%`, multiplica a incerteza base por fator \(> 1\) (`DIR_ACC_PENALIDADE_K = 10`). Isso alimenta `omega_ajustado` / `omega_base` nos artefatos agregados.

#### Artefatos em `criacao_modelo_xgb/`

Por ticker (e frequentemente por horizonte):

| Arquivo | Conteúdo |
|---------|----------|
| `TICKER_predicoes.csv` | `date, ticker, close, y_real, y_pred, erro, fold` |
| `TICKER_h{5,10,21}d_*` | Modelos, params, importância, SHAP, plots por horizonte |
| `TICKER_model.pkl` / `*_clf.pkl` | Modelo vencedor |
| `TICKER_scaler_y.pkl` | Scaler do target |
| `metricas_vencedores.csv` | Horizonte e tipo eleitos + métricas OOF |
| `q_vetor.csv` / `omega_base.csv` | Snapshots agregados |

### 6.4 Horizontes vencedores (snapshot `metricas_vencedores.csv`)

| Ticker | Horizonte | Tipo vencedor | dir_acc OOF (aprox.) | Nota |
|--------|-----------|---------------|----------------------|------|
| ABEV3 | 5 | classificador | ~51.9% | |
| B3SA3 | 21 | regressor | ~53.2% | |
| BBAS3 | 10 | classificador | ~51.7% | |
| BBDC4 | 10 | regressor | ~54.5% | |
| CMIG4 | 21 | regressor | ~62.9% | forte |
| EQTL3 | 21 | regressor | ~51.1% | |
| GGBR4 | 21 | classificador | ~53.2% | |
| ITUB4 | 21 | classificador | ~52.6% | |
| LREN3 | 10 | regressor | ~53.2% | |
| MGLU3 | 21 | regressor | ~59.1% | alta vol / MAE alto |
| PETR3 | 21 | regressor | ~66.1% | forte |
| PETR4 | 21 | regressor | ~68.2% | forte |
| PRIO3 | 21 | regressor | ~56.5% | |
| RADL3 | 21 | regressor | ~57.2% | |
| RAIL3 | 21 | regressor | ~52.2% | |
| SANB11 | 21 | regressor | ~52.6% | |
| SBSP3 | 21 | classificador | ~64.9% | forte |
| SUZB3 | 21 | regressor | ~48.2% | abaixo de 50%; penalizado |
| VALE3 | 10 | regressor | ~47.5% | fraco; sensibilidade |
| WEGE3 | 21 | classificador | ~58.9% | |

**Observação metodológica:** R² OOF em geral próximo de zero ou negativo — típico em previsão de retorno diário/semanal. O projeto prioriza **sinal direcional**, não poder explicativo linear.

Default se ticker ausente em métricas: **`DEFAULT_VIEW_HORIZON = 21`**.

---

## 7. Pipeline B — notícias e sentimento

**Pasta:** `intel/data_acquisition/`

### 7.1 Fluxo

```text
Google News RSS (pt-BR, país BR)
  → noticias_b3.json
  → limpa_duplicados.py          (título)
  → limpa_noticias.py            (exige menção ticker/empresa)
  → analise_sentimento.py        (OpenAI)
  → noticias_b3_sem_duplicados_sentimento.json   ★ entrada do BL
```

Arquivos intermediários:

| Arquivo | Papel |
|---------|--------|
| `noticias_b3.json` | Bruto da coleta RSS |
| `noticias_b3_jobs_done.json` | Checkpoint de jobs |
| `noticias_b3_sem_duplicados.json` | Após dedup |
| `noticias_b3_sem_duplicados_com_link_real.json` | Após resolução Playwright (opcional) |
| `noticias_b3_sem_duplicados_sentimento.json` | ★ Consumido pelo BL |

### 7.2 Premissas e parâmetros da coleta

| Parâmetro | Valor |
|-----------|-------|
| Tickers | Mesmos 20 do BL |
| `COLLECTION_START` / `END` | 2023-01-01 … 2025-12-31 |
| Idioma / país | pt-BR / BR |
| `RSS_MAX_WORKERS` | 2 (estabilidade; evita 503) |
| `RSS_PAUSE_SECONDS` | 4 |
| `RSS_MAX_RETRIES` | 4 |
| Checkpoint | a cada 25 jobs |

Filtros de título (`TICKER_TITLE_TERMS`): exige menção ao código ou nome da empresa (ex.: “Vale”, “Itaú”, “Petrobras”). Há termos legados para ITUB3/BBDC3.

### 7.3 Classificação de sentimento (`analise_sentimento.py`)

- Modelo: **`SENTIMENT_MODEL = "gpt-5.6-luna"`**
- Classes: **positivo / negativo / neutro**
- Prompt: classificar o impacto da manchete no preço da ação `{ticker}`
- Workers: 8
- Cache por chave `ticker||titulo` (reusa classificações já salvas; evita re-pagar API)
- Credencial: `OPENAI_API_KEY` via `.env` (`python-dotenv`)

### 7.4 Agregação diária no BL (`sentiment.py`)

1. Lê o JSON de sentimento.
2. Filtra por **chave de empresa** (4 letras das letras do ticker) quando `match_company=True` — assim PETR3/PETR4 compartilham notícias da Petrobras.
3. Mapeia labels: `{positivo: +1, negativo: −1, neutro: 0}`.
4. Agrega por `(ticker, view_date)`:
   - `sentiment_raw` = média dos scores
   - `news_count` = quantidade de manchetes
5. Sinal suavizado:

\[
\mathrm{sentiment\_signal}
  = \tanh\bigl(\mathrm{sentiment\_raw} \cdot \log(1 + n_{\mathrm{news}})\bigr)
\]

Mais notícias amplificam o sinal, com saturação via \(\tanh\).

Corpus vigente: **7.835** manchetes brutas, **4.681** classificadas (2.418 positivas, 1.253 negativas, 1.010 neutras). Publicação: 2023 = 826, 2024 = 1.374, 2025 = 2.480. Mais volume em VALE3 (826), BBAS3 (707) e PETR3 (432). No Q de 2025, 73,3% dos dias-ativo têm `news_count = 0` e 1,5% têm n ≥ 5.

Scripts auxiliares: `traduz_link_do_rss.py` (Playwright), `extrair_search_xml.py`, `ordenar_noticias_por_data.py`.

---

## 8. Pipeline C — Black-Litterman

**Pasta:** `intel/Blacklitterman/`  
**Entry:** `python pipeline.py`

### 8.1 Etapa 1 — Vetor Q (`xgb_views.py` + `sentiment.py`)

Para cada ativo e dia no período de views (`PERIOD_START`/`PERIOD_END` = **202501–202512**):

1. Localiza `TICKER_predicoes.csv` (ou fallback `*_h21d_predicoes.csv`).
2. Lê `y_pred` → `ret_pred`, `y_real` → `ret_real` (escala do horizonte \(h\)).
3. Resolve \(h\) via `metricas_vencedores.csv` (default 21).
4. Calcula escala de retorno para o sentimento:

\[
\mathrm{ret\_scale}_t
  = \mathrm{mean}\bigl(|r_{\mathrm{real}}|\bigr)_{t-14:t-1}
\]

(fallback: mediana ou `RET_SCALE_FALLBACK = 0.005`).

5. `sentiment_component = sentiment_signal × ret_scale`.
6. Aplica **gate de cobertura** (`NEWS_COUNT_MIN = 5`):

\[
c = \min\bigl(1,\ n_{\mathrm{news}} / \mathrm{NEWS\_COUNT\_MIN}\bigr)
\]

\[
Q
  = \bigl[\alpha + (1-\alpha)(1-c)\bigr]\, r_{\mathrm{pred}}
  + (1-\alpha)\, c \cdot (\mathrm{sent} \times \mathrm{ret\_scale})
\]

| Caso | Comportamento |
|------|----------------|
| Sem notícias (\(c=0\)) | \(Q = r_{\mathrm{pred}}\) (puro XGB) |
| Cobertura cheia (\(c=1\)) | Híbrido clássico \(\alpha r_{\mathrm{pred}} + (1-\alpha)\mathrm{sent}\) |
| Parcial | Interpolação entre os dois |

\(\alpha = \texttt{ALPHA\_Q} = 0{,}386251\) (valor do walk-forward k\*; um Optuna anterior tinha ≈ 0,424 e docs ainda mais antigos citavam 0,50).

7. Alinha à escala diária do prior:

\[
Q_{\mathrm{daily}} = \frac{Q_h}{h}
\]

(`align_q_to_daily`; guarda `Q_horizon` e marca `q_time_scale='daily'`).

**Saída:** `outputs/pipeline/bl_hibrido_q_long.csv`

### 8.2 Etapa 2 — Omega Ω (`omega_calc.py`)

Por ticker/dia:

| Peça | Regra |
|------|--------|
| Erro absoluto | \(\|r_{\mathrm{pred}} - r_{\mathrm{real}}\|\) |
| Escala de erro | Mediana rolling (`ERR_WINDOW=18`, `ERR_MIN_PERIODS=9`), *shift* 1 |
| Confiança do modelo | \(\exp(-\mathrm{erro}/\mathrm{error\_scale})\) |
| Confiança de notícias | \(1 - \exp(-n/3)\) |
| Confiança da view | \(w_m C_m + w_n C_n\), clip \([\texttt{FLOOR},\texttt{CAP}]\) |
| \(\sigma_{ii}\) diário | \(\mathrm{Var}(r_{\mathrm{real}})/h^2\) |
| Base | \(\omega_{\mathrm{base}} = \tau \cdot \sigma_{ii}\) |
| Omega | \(\omega = \omega_{\mathrm{base}} \cdot (1-C)/C\) |
| Boost | Se \(n < 5\), multiplica por `NEWS_OMEGA_BOOST = 3.79598` |
| Hard prior | `HARD_PRIOR_WHEN_LOW_NEWS=True` (confiança vai a ε; view ignorada) |

Pesos atuais: \(w_m = 0{,}659122\), \(w_n = 0{,}340878\); floor = 0,0575; cap = 0,9372.

**Interpretação:** \(\Omega\) alto → view pesa pouco no posterior; \(\Omega\) baixo → puxa \(\mu\) com força.

**Saídas:** `bl_hibrido_omega_long.csv`, `bl_hibrido_q_omega_long.csv`

### 8.3 Etapa 3 — Prior π (`prior.py`)

1. Pesos de mercado \(w_m\) via **market cap** (yfinance `Ticker.info["marketCap"]`).
2. Retornos log diários a partir de `dados_diarios/` no intervalo do prior.
3. \(\Sigma = \mathrm{Cov}(R)\).
4. Taxa livre de risco diária equivalente a `RISK_FREE_ANNUAL = 0.15`:

\[
r_f^{\mathrm{daily}} = (1 + 0.15)^{1/252} - 1
\]

5. Aversão a risco implícita:

\[
\delta = \frac{\mathbb{E}[R_m] - r_f}{\mathrm{Var}(R_m)},
\quad R_m = R\, w_m
\]

6. Prior de equilíbrio:

\[
\pi = \delta\, \Sigma\, w_m
\]

7. Incerteza do prior: \(\tau\Sigma\), com \(\tau = 0{,}0147235\).

**Prior rolling** (`USE_ROLLING_PRIOR=True`): a cada dia de view, reestima \(\pi\)/\(\Sigma\) com histórico **estritamente anterior** ao dia (`index < view_date`), mínimo `ROLLING_MIN_OBS = 80`. Evita look-ahead no prior.

**Saídas:** `bl_hibrido_pi.csv`, `bl_hibrido_sigma.csv`, `bl_hibrido_prior_cov_tau_sigma.csv`

### 8.4 Etapa 4 — Posterior μ

Views absolutas: matriz \(P\) com 1 na coluna do ativo da view (e 0 nas demais). \(\Omega\) diagonal.

\[
\mu
  = \Bigl[(\tau\Sigma)^{-1} + P^\top \Omega^{-1} P\Bigr]^{-1}
    \Bigl[(\tau\Sigma)^{-1}\pi + P^\top \Omega^{-1} Q\Bigr]
\]

Implementação: `np.linalg.pinv` nas matrizes envolvidas.

Pesos brutos (mean-variance com aversão \(\delta\)):

\[
w_{\mathrm{raw}} \propto (\delta\Sigma)^{-1}\mu
\]

depois normalizados para somar 1 (quando finitos).

**Saídas:** `bl_hibrido_posterior_mu.csv`, `bl_hibrido_posterior_weights.csv`

Metadados por dia incluem `views_count`, `view_tickers`, `Q_mean`, `Omega_mean`, `prior_obs`, `prior_end_date`.

### 8.5 Etapa 5 — Controles e rebalanceamento

#### Projeção long-only com teto (`weights.py`)

Algoritmo iterativo:

1. Zera scores negativos.
2. Propõe pesos proporcionais aos scores no orçamento restante.
3. Satura ativos que excedem `max_w` e redistribui o resto.
4. Renormaliza para somar 1.

Invariante: precisa `max_w * n_ativos >= 1` (viabilidade do simplex com teto).

| Controle | Valor atual |
|----------|-------------|
| `LONG_ONLY` | `True` |
| `MAX_WEIGHT_PER_ASSET` | **0,341677** (~34,2%) |

#### Modos de rebalanceamento

| Modo | Comportamento |
|------|----------------|
| `daily` | Ajusta pesos todo dia útil |
| `weekly` | Congela pesos até sexta da semana (`W-FRI`) |
| `monthly` | Congela até o fim do mês |

**Saídas:**  
`bl_hibrido_posterior_weights_controlled.csv`  
`..._controlled_daily.csv` / `_weekly.csv` / `_monthly.csv`

### 8.6 Scripts standalone (debug)

| Script | Função |
|--------|--------|
| `Q.py` | Views Q para um ticker (padrão ABEV3) |
| `omega.py` | Ω a partir do Q standalone |
| `PI.py` | Prior π com retornos diários |
| `PI_incerteza.py` | Prior π com VWAP intradía (`tickers_data/`) |

Saídas em `outputs/standalone/{q,omega,prior}/`.

---

## 9. Avaliação, score e baselines

### 9.1 Backtest (`backtest.py` / `plot_ganho_estimado.py`)

Premissas de execução:

| Premissa | Valor |
|----------|-------|
| Capital inicial | **R$ 100.000** |
| Lag de execução | **1 dia útil** (`EXECUTION_LAG_DAYS`) |
| Custo de transação | **10 bps** × turnover \(0.5\|\Delta w\|_1\) |
| Benchmark | Selic **15% a.a.** capitalizada |
| Dias úteis/ano | 252 |

Séries geradas:

- Capital **estimado** (usando \(\mu\) como retorno esperado do dia)
- Capital **realizado** (retornos log reais dos preços)
- Capital **Selic**

Arquivos: `outputs/backtest/bl_hibrido_ganho_series_*.csv`, `bl_hibrido_ganho_comparativo.csv`  
Figura: `outputs/figures/bl_hibrido_ganho_estimado.png`

### 9.2 Score composto (`score_modelo.py`)

\[
S = 0.35\, S_{\mathrm{pred}} + 0.45\, S_{\mathrm{port}} + 0.20\, S_{\mathrm{risk}}
\quad \in [0, 100]
\]

| Componente | Peso | Conteúdo |
|------------|------|----------|
| Predição | 35% | Hit ratio direcional + erro absoluto (MAE) |
| Portfólio | 45% | Retorno acumulado + Sharpe − drawdown |
| Risco | 20% | Exposição bruta / concentração de pesos |

Saídas: `bl_hibrido_score_summary.csv`, score mensal, por ticker, por mês, resumo de rebalance.

### 9.3 Comparação com baselines (`comparar_bl_vs_markowitz.py`)

Baselines típicos:

- **Markowitz rolling** (janela 60, mín. 40 obs, ridge \(10^{-6}\)), mesmos controles long-only/teto
- Equal-weight / prior-only (conforme implementação do script)
- Selic

Também compara custo **0 vs 10 bps** (`bl_vs_baselines_custo_0_vs_10bps.csv`).

Figuras: `bl_vs_markowitz_plot.png`

---

## 10. Otimização de hiperparâmetros (Optuna)

**Script:** `otimizar_parametros.py`  
**Dataclass:** `BLParams` em `params.py`

### Splits

| Split | Período | Papel |
|-------|---------|--------|
| Train (contexto) | 2023-05 → 2024-12 | Contexto histórico |
| **Validação** | **2024** | Objetivo do Optuna (Sharpe realizado) |
| **Teste** | **2025** | Relatório congelado (não escolhe params) |

### Parâmetros tunáveis

- `alpha_q`, `tau`, `max_weight_per_asset`
- `w_model_conf` / `w_news_conf` (normalizados para somar 1)
- `confidence_floor`, `confidence_cap`
- `err_window`, `err_min_periods`, `rolling_min_obs`

No Optuna clássico (`otimizar_parametros.py`) o gate de notícias fica fixo. No pacote vigente (`melhorias_bl_wf.py`) `news_count_min`, `news_omega_boost` e `hard_prior_when_low_news` foram escolhidos junto com o k\* e gravados no `config.py`.

Defaults: `OPT_DEFAULT_TRIALS = 40`, `OPT_DEFAULT_REBALANCE_MODE = "monthly"`.

Uso:

```bash
cd intel/Blacklitterman
python otimizar_parametros.py
python otimizar_parametros.py --trials 60 --rebalance monthly
python otimizar_parametros.py --apply-best
```

Saídas: `outputs/optimize/best_params.csv`, `trials.csv`, `optimization_summary.json`.

`aplicar_melhores_params.py` pode gravar os best params de volta em `config.py` e reavaliar.

---

## 11. Análises de sensibilidade

### 11.1 Universo (`sensibilidade_universo.py`)

- Roda `run_pipeline` com universo full.
- Roda novamente sem `ATIVOS_SENSITIVITY_DROP` (ex.: VALE3).
- Compara métricas de backtest no modo de rebalance configurado.
- Grava `outputs/score/sensibilidade_universo.csv`.

### 11.2 Gate de notícias (`_check_news_gate.py`)

Self-check das regras de cobertura / boost de Ω — útil como verificação de invariantes, não como bateria de testes unitários completa.

### 11.3 Custos de transação

Comparativo 0 bps vs 10 bps nos baselines (`OUT_MKZ_COST_COMPARE`).

### 11.4 Sensibilidade ao limiar mínimo de notícias (reexecução local em 2025)

Além do limiar histórico de cinco notícias, foi realizada uma comparação exploratória dos cenários `n = 0` a `n = 5`, mantendo os demais hiperparâmetros do modelo e adotando **rebalanceamento mensal**. O experimento foi executado por `simular_cenarios_noticias.py`, conforme os CSVs de saída do pacote de 02/11/2026.

O critério `news_count >= n` é aplicado a cada **par ativo-dia**, e não à soma de notícias do mercado num dia. Há 5.000 observações, correspondentes a 20 ações em 250 datas de views de 2025. Uma mesma notícia pode ser associada a mais de um ticker (ex.: classes de ações da mesma empresa). Por isso, 2.506 é a soma das **ocorrências atribuídas a pares ativo-dia**, não necessariamente o total de notícias únicas.

Para distinguir diferentes conceitos de cobertura, definem-se:

- **Percentual de pares ativo-dia elegíveis:** quantidade de pares com `news_count >= n` dividida por 5.000 (inclui pares sem notícias no denominador);
- **Percentual de pares com mais de `n` notícias:** quantidade de pares com `news_count > n` dividida por 5.000 (critério estrito);
- **Percentual de ocorrências de notícias dentro do padrão:** soma de `news_count` apenas nos pares elegíveis dividida por 2.506;
- **Percentual de dias com algum ativo elegível:** quantidade de datas em que ao menos um ticker atingiu `news_count >= n` dividida por 250;
- **Percentual de dias com total de ocorrências suficiente:** datas em que a soma de `news_count` em todos os tickers atingiu `n`, divididas por 250. Essa estatística **não** representa o gate efetivo aplicado individualmente a cada ticker.

O cenário `n = 0` considera elegíveis todos os 5.000 pares pelo critério inclusivo, inclusive os sem notícias; ele não cria notícias inexistentes. Os limiares `>= n` e `> n` não devem ser tratados como sinônimos.

### 11.3 Custos de transação

Comparativo 0 bps vs 10 bps nos baselines (`OUT_MKZ_COST_COMPARE`).

---

## 12. Premissas (lista explícita)

### Teóricas

1. O mercado possui um equilíbrio implícito recuperável via CAPM reverso (\(\pi = \delta\Sigma w_m\)).
2. Views do investidor podem ser expressas como retornos esperados **absolutos** por ativo (\(P\) diagonal unitária).
3. A confiança nas views é quantificável e entra via \(\Omega\) **diagonal** (views independentes condicionalmente).
4. O investidor é **long-only**, sem alavancagem (exposição bruta ≈ 1 após projeção).
5. A taxa livre de risco pode ser tratada como constante no período (proxy Selic 15% a.a.).

### De dados

6. Preços de fechamento diários bastam para \(\Sigma\), \(\pi\) e backtest principal.
7. Predições OOF walk-forward são a proxy correta de views “sem look-ahead” no treino XGB.
8. Manchetes RSS + classificação GPT em 3 classes capturam viés qualitativo útil.
9. Agrupar tickers pela empresa (PETR3/PETR4, ITUB3/ITUB4 legado, etc.) é adequado para sentimento.
10. Market caps pontuais do yfinance são proxy aceitável de \(w_m\) (podem ser quase estáticos no momento do run).

### De implementação / risco

11. Execução com **1 dia de lag** e **10 bps** aproxima fricção real sem modelar impacto de mercado.
12. Rebalance **monthly** é o modo privilegiado no tuning vigente (pode divergir do melhor modo ex-post).
13. Gate mínimo de **5 notícias** e hard prior evitam sentimento esparso dominar \(Q\); abaixo disso a view é ignorada e \(\Omega\) é multiplicado por ≈ 3,80.
14. Conversão \(Q_h/h\) e \(\Omega/h^2\) alinha escalas multi-horizonte ao prior diário (aproximação de retornos i.i.d. no horizonte).
15. Hold-out 2024 (val) / 2025 (teste) evita escolher hiperparâmetros no mesmo ano do relatório final.
16. Decaimento amostral \(\lambda=5\) no treino XGB prioriza observações recentes.
17. Acurácia direcional, não R², é o critério de seleção de horizonte/modelo.

---

## 13. Parâmetros oficiais (config atual)

Valores em `intel/Blacklitterman/config.py`, gravados a partir de `melhorias_bl_wf_best.json` (walk-forward, k\* = 5). Não são os de `optimization_summary.json`.

**Separação entre versões:** no `config.py` do pacote local enviado em 02/11/2026, `NEWS_COUNT_MIN = 1`, `HARD_PRIOR_WHEN_LOW_NEWS = True`, `NEWS_OMEGA_BOOST = 3.79598`, `ALPHA_Q = 0.386251` e `TAU = 0.0147235`. O arquivo histórico `outputs/optimize/melhorias_bl_wf_best.json` ainda registra `news_count_min = 5`. Assim, a tabela abaixo documenta o **baseline publicado anteriormente**; a alteração local para `n = 1` deve ser registrada como experimento posterior, e não como resultado da seleção walk-forward de 2024.

### Views (Q)

| Parâmetro | Valor |
|-----------|-------|
| `PERIOD_START` / `PERIOD_END` | `202501` / `202512` |
| `ALPHA_Q` | **0.386251** |
| `RET_SCALE_WINDOW` | 14 |
| `RET_SCALE_MIN_PERIODS` | 5 |
| `RET_SCALE_FALLBACK` | 0.005 |
| `DEFAULT_VIEW_HORIZON` | 21 |
| `SENTIMENT_MAP` | pos=1, neg=−1, neu=0 |

### Prior

| Parâmetro | Valor |
|-----------|-------|
| `PRIOR_START_DATE` / `END` | 2023-01-01 / 2025-12-31 |
| `RISK_FREE_ANNUAL` / `SELIC_ANNUAL` | 0.15 |
| `TAU` | **0.0147235** |
| `USE_ROLLING_PRIOR` | True |
| `ROLLING_MIN_OBS` | 80 |

### Omega

| Parâmetro | Valor |
|-----------|-------|
| `ERR_WINDOW` | 18 |
| `ERR_MIN_PERIODS` | 9 |
| `CONFIDENCE_FLOOR` | 0.0575035 |
| `CONFIDENCE_CAP` | 0.937247 |
| `W_MODEL_CONF` | 0.659122 |
| `W_NEWS_CONF` | 0.340878 |
| `NEWS_COUNT_MIN` | 5 |
| `NEWS_OMEGA_BOOST` | 3.79598 |
| `HARD_PRIOR_WHEN_LOW_NEWS` | True |
| `EPS` | 1e-8 |

### Portfólio e backtest

| Parâmetro | Valor |
|-----------|-------|
| `LONG_ONLY` | True |
| `MAX_WEIGHT_PER_ASSET` | **0.341677** |
| `INITIAL_CAPITAL_BRL` | 100_000 |
| `EXECUTION_LAG_DAYS` | 1 |
| `TRANSACTION_COST_BPS` | 10 |
| `TRADING_DAYS_YEAR` | 252 |

### Optuna (hold-out)

| Parâmetro | Valor |
|-----------|-------|
| `OPT_TRAIN_START` / `END` | 202305 / 202412 |
| `OPT_VAL_START` / `END` | 202401 / 202412 |
| `OPT_TEST_START` / `END` | 202501 / 202512 |
| `OPT_DEFAULT_TRIALS` | 40 |
| `OPT_DEFAULT_REBALANCE_MODE` | monthly |

### Treino XGB (`intel/model_training/config.py` + flags em `train_xgb.py`)

| Parâmetro | Valor |
|-----------|-------|
| `DAILY_START_DATE` | 2018-01-01 |
| `FEATURE_WINDOWS` | [3, 5, 14, 21, 42, 63] |
| `HORIZONS` | [5, 10, 21] |
| `OPTUNA_TRIALS` (treino) | 40 |
| `USE_CLASSIFICATION` | True |
| `USE_SHAP` | True |

### Notícias (`intel/data_acquisition/config.py`)

| Parâmetro | Valor |
|-----------|-------|
| `SENTIMENT_MODEL` | gpt-5.6-luna |
| `SENTIMENT_MAX_WORKERS` | 8 |
| `RSS_MAX_WORKERS` | 2 |
| `RSS_PAUSE_SECONDS` | 4 |
| Coleta | 2023-01-01 … 2025-12-31 |
| RSS bruto | 7.835 |
| Sem duplicata + sentimento | 4.681 |
| Positivo / negativo / neutro | 2.418 / 1.253 / 1.010 |
| Por ano (publicação) | 2023 = 826, 2024 = 1.374, 2025 = 2.480 |
| Dias-ativo em Q 2025 sem notícia | 73,3% |
| Dias-ativo com n ≥ 5 | 1,5% |

---

## 14. Resultados materializados (snapshots)

> Os números abaixo são **snapshots de arquivos no repositório**. Reexecuções podem divergir (yfinance, market cap, notícias, seed do Optuna). Cite o arquivo junto com o número. **Não misture experimentos.**

### 14.1 Resultado vigente — walk-forward k\* (`melhorias_bl_wf_best.json`)

Params iguais ao `config.py`. k escolhido na validação 2024 = **5** (score 1,70). Teste 2025, 249 dias, capital inicial R$ 100 mil, subconjunto top-5:

| Estratégia | Sharpe | Retorno | Capital final | Max drawdown |
|------------|--------|---------|---------------|--------------|
| BL no top-5 | 0,85 | +15,2% | R$ 114,0 mil | −9,6% |
| Híbrido (rank XGB + Markowitz) | 0,40 | +7,4% | R$ 106,0 mil | −11,3% |
| Markowitz clássico no mesmo top-5 | −0,12 | −2,5% | R$ 96,2 mil | −18,4% |

Fonte: `outputs/score/melhorias_bl_wf_kstar_metricas.csv`. Figura: `outputs/figures/melhorias_bl_wf_kstar_capital.png`.

No sweep k = 1…20 (`melhorias_bl_wf_oos_sweep.csv`) outros k ficam melhores fora da amostra (híbrido em k = 17 chega a Sharpe ≈ 2,04). Esse número **não** é o resultado principal: o k foi escolhido antes, e o escolhido é 5.

### 14.2 Optuna anterior (`optimization_summary.json`) — não é a config atual

40 trials, rebalance monthly, α ≈ 0,424, τ ≈ 0,011, teto ≈ 0,427, gate de 2 notícias, hard prior desligado.

| Split | Sharpe | Retorno | Capital final | Max drawdown |
|-------|--------|---------|---------------|--------------|
| Val 2024 | 1,31 | +21,3% | R$ 148,6 mil | −8,4% |
| Teste 2025 | 1,82 | +30,5% | R$ 208,9 mil | −8,9% |

O CSV `markowitz/bl_vs_markowitz_resumo.csv` (10 bps, 249 dias) **não reproduz** os R$ 208,9 mil. Nele, o monthly do BL fecha em R$ 130,8 mil (Sharpe 1,93) contra Markowitz R$ 132,4 mil (Sharpe 1,89), equal-weight R$ 125,9 mil e Selic R$ 114,8 mil. O weekly e o daily desse CSV perdem para Markowitz. Tratar 14.2 como rodada antiga, não como o backtest da config vigente.

### 14.3 Baseline histórico (`outputs/baseline_2026-09-08.json`)

Label `baseline_pre_mudancas`. Universo e params anteriores a esta versão. Não usar no capítulo de resultados da versão atual.

### 14.4 Figuras

- `figures/melhorias_bl_wf_kstar_capital.png` — resultado vigente
- `figures/melhorias_bl_wf_oos_sweep.png`
- `figures/bl_hibrido_ganho_estimado.png`
- `figures/bl_vs_markowitz_plot.png` — Optuna anterior
- `figures/sensibilidade_tamanho_universo.png` — gerada com teto ≈ 0,427
- Plots por ticker em `criacao_modelo_xgb/`

---

### 14.5 Reexecução exploratória do gate de notícias: `n = 0` a `n = 5` (pacote de 02/11/2026)

Este experimento utiliza os **20 ativos** em 2025, com **rebalanceamento mensal**, um dia de defasagem de execução e custo de turnover configurado em 10 bps. São **5.000 pares ativo-dia** (20 tickers × 250 datas de Q), dos quais 1.334 têm ao menos uma notícia atribuída e 3.666 não têm nenhuma (respectivamente, **26,68%** e **73,32%**). O total de ocorrências `news_count` na matriz Q é **2.506**, com possibilidade de duplicação por ativo. Houve notícia atribuída em 249 das 250 datas (99,6% dos dias).

**Cobertura da informação por limiar:**

| n | Pares ativo-dia com ≥ n | % pares elegíveis | % pares com > n | % ocorrências de notícias em pares elegíveis | % dias com algum ativo ≥ n |
|---:|---:|---:|---:|---:|---:|
| 0 | 5.000 | 100,00% | 26,68% | 100,00% | 100,0% |
| 1 | 1.334 | 26,68% | 10,66% | 100,00% | 99,6% |
| 2 | 533 | 10,66% | 5,24% | 68,04% | 80,8% |
| 3 | 262 | 5,24% | 2,64% | 46,41% | 57,6% |
| 4 | 132 | 2,64% | 1,48% | 30,85% | 38,0% |
| 5 | 74 | 1,48% | 1,08% | 21,59% | 23,6% |

A tabela de pares elegíveis é reproduzida diretamente de `outputs/score/cenarios_noticias_percentuais.csv`. As colunas de **ocorrências** e **dias** foram calculadas a partir do arquivo `outputs/pipeline/bl_hibrido_q_long.csv` presente no mesmo pacote. As porcentagens de notícias não se referem a notícias únicas, mas a ocorrências associadas a tickers e datas.

**Desempenho dos seis cenários, com os demais parâmetros mantidos:**

| n | Score total (0–100) | Score de previsão | Score de portfólio | Retorno realizado | Sharpe realizado | Máx. drawdown |
|---:|---:|---:|---:|---:|---:|---:|
| 0 | 73,35 | 33,07 | 92,82 | +31,58% | 1,83 | −10,76% |
| **1** | **73,92** | **33,07** | **94,11** | **+44,99%** | **2,31** | **−8,83%** |
| 2 | 73,43 | 33,07 | 93,02 | +36,91% | 1,90 | −10,47% |
| 3 | 73,44 | 33,07 | 93,04 | +27,40% | 1,56 | −10,44% |
| 4 | 73,44 | 33,07 | 93,04 | +25,06% | 1,50 | −10,44% |
| 5 | 73,44 | 33,07 | 93,04 | +22,81% | 1,37 | −10,44% |

Em todos os cenários, o score de risco é **100,00**; o score de previsão é idêntico (**33,072**) porque a base de previsões quantitativas permanece a mesma. O score composto varia relativamente pouco entre os limiares, apesar das diferenças substanciais de retorno e Sharpe, em parte devido às funções de normalização e saturação do `score_modelo.py`.

O cenário `n = 1` apresentou os melhores indicadores desta **comparação retrospectiva de 2025**: score **73,92**, retorno acumulado **44,99%**, Sharpe **2,31** e drawdown máximo **−8,83%**. Na série mensal correspondente do arquivo `outputs/backtest/bl_hibrido_ganho_series_monthly.csv`, o capital final foi **R$ 144.994,98** para capital inicial de R$ 100 mil (249 observações após o lag de execução). Em relação ao cenário `n = 5` no mesmo sweep, houve diferença de aproximadamente **22,18 pontos percentuais de retorno** e **0,93 ponto de Sharpe**.

**Interpretação e cautelas:** a redução do limiar de 5 para 1 aumentou a cobertura de 1,48% para 26,68% dos pares ativo-dia. O melhor desempenho observado para `n = 1` **não prova superioridade preditiva fora da amostra**: a escolha do melhor `n` foi feita olhando os seis resultados de 2025, que também são os resultados reportados. Para considerar `n = 1` como hiperparâmetro oficial, seria necessário escolhê-lo numa janela de validação anterior e testá-lo em período temporal posterior, sem nova seleção ex-post.

**Não misturar universos e rodadas:** este sweep usa os 20 tickers, enquanto o experimento histórico da seção 14.1 utiliza o subconjunto top-5 (`k* = 5` é **quantidade de ativos**, e não quantidade de notícias). Além disso, os arquivos `outputs/markowitz/bl_vs_markowitz_resumo.csv` e `outputs/score/melhorias_bl_wf_kstar_metricas.csv` no pacote têm data anterior à reexecução local e não são comparações atualizadas contra os seis cenários. O `bl_hibrido_score_summary.csv` reporta o modo **daily** (score 61,05; retorno 10,30%; Sharpe 0,58) e não deve ser confundido com o modo **monthly** (score 73,92; retorno 44,99%; Sharpe 2,31) utilizado no sweep.

**Arquivos-fonte:** `outputs/score/cenarios_noticias_percentuais.csv`, `outputs/score/cenarios_noticias_scores.csv`, `outputs/pipeline/bl_hibrido_q_long.csv`, `outputs/score/bl_hibrido_rebalance_resumo.csv`, `outputs/backtest/bl_hibrido_ganho_series_monthly.csv` e `config.py`, conforme arquivo `Blacklitterman.zip` fornecido em 02/11/2026. Os valores foram conferidos nos arquivos gerados; não foi executado um novo backtest nesta atualização textual.

---

## 15. Ordem de execução e dependências

### 15.1 Ambiente

```bash
python -m venv venv
venv\Scripts\activate          # Windows
pip install -r requirements.txt
```

Arquivo `.env` na raiz (apenas para sentimento):

```
OPENAI_API_KEY=sk-...
```

### 15.2 Comandos

```bash
# A) Treino XGB (quando for regenerar Q)
cd intel/model_training
python pipeline.py                 # download + features + train
# ou: python pipeline.py download features
# ou: python download_daily.py && python prepare_features.py && python train_xgb.py

# B) Notícias (quando for regenerar sentimento)
cd intel/data_acquisition
python cata_link_rss.py
python limpa_duplicados.py
python limpa_noticias.py
python analise_sentimento.py

# C) Black-Litterman + avaliação
cd intel/Blacklitterman
python pipeline.py
python plot_ganho_estimado.py
python score_modelo.py
python comparar_bl_vs_markowitz.py

# Pesquisa / TCC
python otimizar_parametros.py [--trials N] [--apply-best]
python sensibilidade_universo.py
```

### 15.3 Dependências (`requirements.txt`)

| Categoria | Pacotes |
|-----------|---------|
| Core | numpy, pandas, scikit-learn, matplotlib, joblib |
| Dados | yfinance |
| ML | xgboost, optuna |
| Sentimento | openai, python-dotenv |
| Notícias | feedparser |
| Opcional | shap, lightgbm, playwright |

---

## 16. Mapa de saídas

```
intel/Blacklitterman/outputs/
├── pipeline/       # Q, Ω, π, Σ, μ, pesos (daily/weekly/monthly)
├── backtest/       # séries de capital
├── score/          # score 0–100 + sensibilidade de universo
├── markowitz/      # BL vs baselines + custos
├── optimize/       # Optuna (trials, best_params, summary)
├── figures/        # PNGs
├── standalone/     # debug Q / Ω / π VWAP
│   ├── q/
│   ├── omega/
│   └── prior/
└── baseline_*.json # snapshots de baseline
```

### Arquivos principais do `pipeline/`

| Arquivo | Conteúdo |
|---------|----------|
| `bl_hibrido_q_long.csv` | Views Q diárias por ticker |
| `bl_hibrido_omega_long.csv` | Ω dinâmico |
| `bl_hibrido_q_omega_long.csv` | Q + Ω unidos |
| `bl_hibrido_pi.csv` | Prior π |
| `bl_hibrido_sigma.csv` | Matriz Σ |
| `bl_hibrido_prior_cov_tau_sigma.csv` | \(\tau\Sigma\) |
| `bl_hibrido_posterior_mu.csv` | Retorno posterior μ |
| `bl_hibrido_posterior_weights*.csv` | Pesos brutos / controlados / rebalanceados |

---

## 17. Limitações e inconsistências documentais

### Limitações metodológicas

1. R² negativo/baixo e acurácia direcional só ligeiramente acima de 50% em vários nomes — o sinal preditivo é frágil.
2. Sentimento baseado só em **título**, não no corpo da notícia.
3. Universo fixo; sem short; sem restrições setoriais explícitas na otimização.
4. Custos lineares simplificados; sem modelagem de liquidez/impacto intradía no backtest principal.
5. Market cap pontual pode não refletir a evolução temporal dos pesos de mercado.
6. Aproximação \(Q/h\) e \(\Omega/h^2\) assume escala temporal simples.
7. Poucos testes automatizados além de `_check_news_gate.py`.
8. Com o gate de 5 notícias, 73,3% dos dias-ativo de 2025 não têm manchete e só 1,5% têm n ≥ 5 — o sentimento quase não altera o portfólio vigente.
9. O melhor k ex-post (sweep 1…20) não é o k\* escolhido em 2024. Citar o k = 5, não o melhor k do teste.

### Cuidados ao citar números

1. Pasta `criacao_modelo_xgb/` contém **artefatos**, não o código de treino (código em `intel/model_training/`).
2. Tickers legados (ITUB3/BBDC3) podem coexistir no JSON; o agrupamento por empresa mitiga isso.
3. `optimization_summary.json`, `bl_vs_markowitz_resumo.csv` e a sensibilidade de tamanho usam um Optuna anterior (teto ≈ 0,427). O resultado da config atual é o k\* da seção 14.1.
4. `score/bl_hibrido_score_summary.csv` e `sensibilidade_universo.csv` (universo de 5) não batem com 14.1. Regenerar depois de `pipeline.py` com a config atual antes de citar score composto.

---

**Nota da reexecução de 02/11/2026:** para o experimento local com `NEWS_COUNT_MIN = 1`, o percentual de pares ativo-dia sem notícias continua em **73,32%**; o conjunto elegível sobe para **26,68%**. A comparação de 2025 favorece `n = 1` nesse recorte, mas não constitui uma validação temporal independente do limiar. Os arquivos de Markowitz e do experimento histórico top-5 não foram regenerados nesta mesma rodada e devem ser identificados como históricos. A função de confiança de `omega_calc.py` também merece auditoria temporal: no código consultado, `model_confidence` é calculada a partir do erro `ret_pred - ret_real` da própria observação, o que requer verificar a disponibilidade efetiva de `ret_real` na data de decisão.

---

## 18. Sugestão de estrutura de capítulos do TCC

1. **Introdução** — problema de alocação na B3; lacuna entre Markowitz e views subjetivas/ML.
2. **Referencial teórico** — Markowitz, CAPM reverso, Black-Litterman, NLP/sentimento, validação walk-forward.
3. **Metodologia** — arquitetura em 3 pipelines; fórmulas \(Q\), \(\Omega\), \(\pi\), \(\mu\); premissas.
4. **Dados** — universo 20 ativos; preços; features IBOV; corpus RSS 2023–2025.
5. **Modelo quantitativo** — XGB, horizontes, classificação vs regressão, métricas OOF.
6. **Camada qualitativa** — coleta, limpeza, GPT, agregação \(\tanh\).
7. **Otimização BL** — prior rolling, gate de notícias, restrições, custos, Optuna hold-out.
8. **Resultados** — k\* = 5 em 2025 vs Markowitz no mesmo subconjunto; sweep de k só como robustez; não misturar com o Optuna anterior.
9. **Discussão** — sinal perto de 50%; sentimento esparso com gate de 5; k escolhido ex-ante vs melhor k ex-post.
10. **Conclusão e trabalhos futuros** — corpo da notícia, short, liquidez, regimes, paper trading.

### Possíveis contribuições a destacar

- Pipeline híbrido **ML + NLP + BL** operacionalizado ponta a ponta para B3.
- Gate de cobertura de notícias acoplado a \(Q\) e \(\Omega\).
- Conversão multi-horizonte (\(h \in \{5,10,21\}\)) para escala diária.
- Prior rolling com embargo temporal.
- Protocolo de hold-out (val 2024 / teste 2025) para hiperparâmetros do BL.
- Avaliação com lag de execução e custos de turnover.

---

## 19. Glossário

| Termo | Significado neste projeto |
|-------|---------------------------|
| \(Q\) | Vetor de views (retorno esperado opinado por ativo/dia) |
| \(\Omega\) | Incerteza (variância) das views |
| \(\pi\) | Prior de equilíbrio de mercado |
| \(\Sigma\) | Matriz de covariância dos retornos |
| \(\tau\) | Escala da incerteza do prior |
| \(\mu\) | Posterior BL (retorno esperado ajustado) |
| \(w\) | Pesos do portfólio |
| \(P\) | Matriz que mapeia views → ativos (aqui: identidade por view absoluta) |
| \(\delta\) | Aversão a risco implícita do mercado |
| OOF | Out-of-fold / walk-forward (predição sem treinar no mesmo trecho) |
| dir_acc | Acurácia direcional (sinal de \(y_{\mathrm{pred}}\) vs \(y_{\mathrm{real}}\)) |
| Long-only | Sem posições vendidas |
| Gate de notícias | Cobertura mínima antes de misturar sentimento em \(Q\) |
| Selic (proxy) | Taxa livre de risco anual 15% usada no backtest |

---

## 20. Referências internas do repositório

| Documento / módulo | Conteúdo |
|--------------------|----------|
| `README.md` | Visão geral e como executar (checar universo) |
| `intel/Blacklitterman/README.md` | Detalhes do módulo BL |
| `intel/Blacklitterman/COMO_FUNCIONA.md` | Guia narrativo do fluxo BL |
| `intel/Blacklitterman/outputs/README.md` | Layout das pastas de saída |
| `intel/Blacklitterman/config.py` | ★ Parâmetros vigentes |
| `intel/model_training/train_xgb.py` | ★ Spec do treino XGB |
| `intel/data_acquisition/config.py` | ★ Spec de notícias |
| `criacao_modelo_xgb/metricas_vencedores.csv` | Horizontes eleitos |
| `intel/Blacklitterman/outputs/optimize/optimization_summary.json` | Resultado Optuna |
| `intel/Blacklitterman/outputs/baseline_2026-09-08.json` | Snapshot de baseline |

---

## Apêndice A — Pesos brutos → pesos controlados (pseudocódigo)

```text
scores = max(w_raw, 0)
se sum(scores) ≈ 0: retorna equal-weight
budget = 1
enquanto budget > ε e há ativos livres:
    proporcione budget pelos scores dos livres
    se algum >= max_w: satura em max_w, remove do livre, reduz budget
    senão: aplica proposta e termina
renormaliza para somar 1
```

## Apêndice B — Checklist de reprodução para o TCC

- [ ] Ambiente Python + `requirements.txt`
- [ ] `.env` com `OPENAI_API_KEY` (só se for reclassificar sentimento)
- [ ] `dados_diarios/` populado (ou `python pipeline.py download features` em `model_training`)
- [ ] Artefatos em `criacao_modelo_xgb/` (ou `train`)
- [ ] JSON `noticias_b3_sem_duplicados_sentimento.json`
- [ ] `cd intel/Blacklitterman && python pipeline.py`
- [ ] `plot_ganho_estimado.py` → `score_modelo.py` → `comparar_bl_vs_markowitz.py`
- [ ] (Opcional) `otimizar_parametros.py` e `sensibilidade_universo.py`
- [ ] Conferir que o texto do TCC cita **20 ativos** e os parâmetros **atuais** do `config.py`

---

*Documento gerado como material de apoio ao TCC a partir da leitura do código e artefatos do repositório. Atualize os snapshots numéricos sempre que reexecutar Optuna/backtest.*
