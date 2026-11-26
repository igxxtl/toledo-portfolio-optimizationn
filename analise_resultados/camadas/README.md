# Resultados intermediários — rodada exploratória de 2025

Reprodução, na raiz do repositório, com Python, numpy e pandas:

```powershell
python -m analise_resultados.camadas_intermediarias
python -m analise_resultados.diagnostico_xgb
python -m analise_resultados.integrar_camadas
```

O último comando integra o fragmento ao TCC em `Documents/TCC 2`. Não altera os resultados financeiros históricos. As variantes compartilham preços, horizonte selecionado em 2024, calendário, covariância, custos, piso/teto, seleção e motor de execução. O manifesto documenta as intervenções e os hashes dos arquivos de entrada. A comparação automática exige que o híbrido reproduza as três trajetórias livres históricas com diferença inferior a 1e-7 ponto percentual.

- `metricas.csv`: cinco variantes, n=5/10/20, 30 subconjuntos comuns para 5/10, um para 20, e seleção própria; 320 simulações.
- `comparacao_pareada.csv`: retornos e diferenças contra o prior no mesmo subconjunto; dispersão não é intervalo de confiança.
- `auditoria.csv`: previsão, sentimento, Q, confiança, Omega, prior e posterior por decisão. `gate_active` indica elegibilidade textual; não indica ativação do XGB livre ou do prior.
- `pesos.csv` e `distancia_pesos_prior.csv`: carteiras-alvo e meia L1 contra prior, pareadas por caso/data. A distância não é turnover realizado.
- `exemplos_cadeia.csv`: exemplos deterministas de decisões sem notícias, com gate e com discordância de sinais quando disponível.
- `xgb_baselines_trimestres.csv`: avaliação anual/trimestral com alvos maturados, retorno zero e média histórica de alvos já conhecidos. Não comparar RMSE entre horizontes como se fossem o mesmo problema.
- `shap_historico_horizontes_selecionados.csv`: explicações históricas; não são explicações exclusivas de 2025.

## Avaliação humana concluída

As 300 notícias foram preenchidas por um único avaliador e importadas do CSV fornecido pelo usuário. As colunas avaliador 1 e consenso coincidem; não houve um segundo avaliador. Concordância observada: 165/300 (55%); macro-F1: 55,69%; kappa: 0,355. `sentimento_avaliacao.json` contém o resumo, `sentimento_matriz_confusao.csv` usa linhas humanas e colunas do modelo, e `sentimento_divergencias.csv` lista os 135 desacordos. `importacao_rotulacao.json` registra origem e hash. O CSV anterior foi preservado com sufixo de hash.

Para incorporar outro arquivo preenchido, validar identidade da amostra e preservar a versão anterior:

```powershell
python -m analise_resultados.avaliar_rotulacao --input CAMINHO_DO_CSV
python -m analise_resultados.integrar_camadas
```

### Instruções de rotulação e reprodução

Preencher `rotulacao_humana.csv` sem abrir `gabarito_modelo.csv`. A amostra é aleatória simples de 300 registros (semente 2025). Os avaliadores classificam a mensagem sobre o impacto na empresa como `positivo`, `negativo` ou `neutro`, sem consultar retornos posteriores. Cada avaliador preenche sua coluna de forma independente; depois registram divergências em `observacao` e preenchem `rotulo_consenso`. Eventos duplicados entre registros podem reduzir a independência da amostra.

```powershell
python -m analise_resultados.avaliar_rotulacao
```

Sem consenso humano, o avaliador encerra sem estimar métricas. A tabela humana não é sobrescrita ao reexecutar a ablação. Não mudar a semente/amostra sem guardar as anotações e seu gabarito correspondente. As falhas técnicas do classificador histórico podem ter sido convertidas em neutro; a distribuição de classes não certifica qualidade da rotulação.

## Alcance

Análise retrospectiva com gate escolhido após observar 2025, estimadores históricos e apenas um ano financeiro. Não demonstra superioridade independente fora da amostra. O controle XGB com gate retira somente o componente textual de Q, mantendo a confiança híbrida. XGB livre muda também gate/confiança. Textual herda escala/horizonte quantitativos. Não atribuir o ganho contra Markowitz automaticamente às views.
