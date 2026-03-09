# Guia do `Q.py` (ABEV3, Lag-Llama, Walk-Forward)

Este documento explica como o script `Q.py` monta a view `Q` para Black-Litterman no teste com um ativo (`ABEV3`), usando:

- serie historica horaria em `tickers_data`
- modelo treinado em `modelo_7h_vwap`
- noticias com sentimento em `intel/data acquisition/noticias_b3_sem_duplicados_sentimento.json`

## 1) Objetivo do script

Gerar uma view diaria de retorno esperado (`Q`) para o ativo, combinando:

1. previsao quantitativa de retorno de 7 horas (1 dia util da B3) via Lag-Llama
2. sinal qualitativo de sentimento de noticias do mesmo dia

O periodo do teste esta configurado no proprio arquivo:

- `PERIOD_START = "202506"`
- `PERIOD_END = "202512"`

## 2) Regras de mercado usadas

O script respeita a logica de pregão da B3:

- somente dias uteis (`segunda` a `sexta`)
- somente barras horarias de `13:00` a `19:00`
- total de `7` barras por dia (equivalente ao horizonte de previsao)

Dias incompletos (sem as 7 barras esperadas) sao ignorados no walk-forward.

## 3) Fluxo passo a passo

### Passo A - Carregar e preparar precos do ABEV3

1. Le o CSV `tickers_data/BMFBOVESPA_DLY_ABEV3, 60.csv`
2. Converte `time` para datetime
3. Filtra apenas horario de pregão B3 (dias uteis, 13h-19h)
4. Calcula VWAP com janela de 7 barras:

`vwap_t = sum(close * volume, janela=7) / sum(volume, janela=7)`

5. Define o alvo do modelo:

`log_ret_vwap_t = ln(vwap_t / vwap_{t-1})`

6. Cria `time_fake` continuo em frequencia horaria (sem buracos) para manter compatibilidade com o modelo treinado.

---

### Passo B - Carregar modelo Lag-Llama

1. Tenta carregar o predictor salvo em `modelo_7h_vwap`
2. Se existir pasta `lag-Llama`, adiciona ao `sys.path` para desserializacao correta

---

### Passo C - Walk-Forward validation (previsao diaria de 7h)

Para cada dia do periodo (`2025-06` a `2025-12`):

1. Separa as 7 barras reais do dia (`13h` a `19h`)
2. Monta contexto com **todo historico anterior ao inicio do dia**
3. Roda o modelo para prever os proximos 7 passos de `log_ret_vwap`
4. Soma os 7 retornos previstos para formar retorno previsto diario:

`ret_7h_pred = sum_{i=1..7}(ret_pred_i)`

5. Soma os 7 retornos reais do mesmo dia:

`ret_7h_real = sum_{i=1..7}(ret_real_i)`

Isso evita vazamento de informacao, pois cada previsao usa apenas passado.

---

### Passo D - Agregar sentimento das noticias no mesmo dia

Para cada noticia de `ABEV3`:

- `positivo = +1`
- `negativo = -1`
- `neutro = 0`

Agrupamento por dia:

1. `sentiment_raw = media(dos scores do dia)`
2. `news_count = quantidade de noticias do dia`
3. sinal final suavizado:

`sentiment_signal = tanh(sentiment_raw * log1p(news_count))`

Esse sinal fica entre `-1` e `+1`.

---

### Passo E - Combinar previsao + sentimento para montar Q

Escala de retorno para traduzir sentimento em unidade financeira:

`ret_scale = media movel de 20 dias de |ret_7h_real| (com shift de 1 dia)`

Componente de noticia:

`sentiment_component = sentiment_signal * ret_scale`

View final:

`Q = alpha * ret_7h_pred + (1 - alpha) * sentiment_component`

No codigo:

- `ALPHA = 0.75` -> 75% modelo, 25% noticias

## 4) Arquivos gerados

## `q_ab_ev3_walkforward_long.csv`

Arquivo detalhado por dia, usado para auditoria e analise.

Colunas:

- `view_date`: data da view diaria (dia util previsto)
- `ticker`: ativo da linha (`ABEV3`)
- `Q`: retorno esperado final do dia (view para BL)
- `ret_7h_pred`: retorno previsto pelo Lag-Llama (soma das 7h)
- `ret_7h_real`: retorno realizado no dia (soma das 7h reais)
- `time_real_inicio`: primeiro timestamp real do bloco diario (13:00)
- `time_real_fim`: ultimo timestamp real do bloco diario (19:00)
- `time_fake_inicio`: timestamp fake correspondente ao inicio do bloco
- `time_fake_fim`: timestamp fake correspondente ao fim do bloco
- `sentiment_signal`: sinal diario de sentimento em [-1, 1]
- `sentiment_component`: impacto de noticias em unidade de retorno
- `news_count`: numero de noticias do ticker naquele dia
- `ret_scale`: escala de volatilidade usada para converter sentimento em retorno

---

## `q_ab_ev3_walkforward_matrix.csv`

Matriz final de views (`Q`) em formato de entrada para otimizacao:

- linhas: `view_date`
- colunas: ativos (neste teste, apenas `ABEV3`)
- valor da celula: `Q` do dia para o ativo

Exemplo conceitual:

- `Q[2025-06-10, ABEV3] = -0.0079` (valor ilustrativo)

## 5) Como interpretar rapidamente

- `Q > 0`: view de alta para o dia
- `Q < 0`: view de baixa para o dia
- `|Q|` maior: conviccao maior
- quando `news_count = 0`, o resultado fica praticamente dominado por `ret_7h_pred`
- quando ha noticias relevantes, `sentiment_component` ajusta o sinal final

## 6) Pontos importantes

- `time` real e usado para respeitar a cronologia de mercado.
- `time_fake` e usado apenas para compatibilidade de frequencia do modelo.
- O walk-forward evita olhar o futuro.
- O formato atual esta preparado para 1 ativo; para multiativos, a mesma logica pode ser replicada por ticker e depois pivotada.
