# Ampliação da base de notícias (25/11/2026)

Scripts de busca, coleta, preparação, gravação e classificação de manchetes usados para montar a base complementar de notícias. Essa base alimenta o teste em [`../backtest_base_noticias_2026_11_25/`](../backtest_base_noticias_2026_11_25/).

- `busca_*`, `coleta_*`, `segunda_coleta_noticias.py`: buscas e coletas por ticker, empresa, veículo, RSS, ADR e idioma, com os CSVs gerados.
- `preparar_*`, `gravar_*`, `testar_arquivos_*`: preparo e gravação dos lotes de 89 e 129 candidatas, da ampliação e da coleta multilíngue.
- `processar_sentimento_b3.py`, `publicar_sentimento_b3.py`, `sentimento_b3/`: classificação de sentimento e rótulos.
- `integrar_noticias_b3.py`, `backtest_noticias.py`, `comparar_backtest_noticias.py`: integração à base e backtest comparativo.

Os scripts foram escritos na pasta `TCC 2/tmp` da máquina de origem, então alguns caminhos relativos podem precisar de ajuste.
