# Simulação financeira do XGBoost conjunto

Comparação retrospectiva de 2025. O pool fixo destacado na avaliação preditiva é aplicado ao mesmo motor financeiro e aos mesmos horizontes da rodada original. Os modelos e arquivos históricos não são substituídos.

## Configurações

- `BL_atual`: híbrido original.
- `BL_pool`: previsão do pool, com confiança calculada a partir de seus próprios erros já maturados.
- `BL_pool_omega_fixo`: previsão do pool, mas matriz Omega original. Isola a alteração da opinião mantendo a incerteza anterior.
- `XGB_atual` / `XGB_pool`: expectativas quantitativas diretamente no mesmo otimizador de média-variância, sem atualização BL. Não são estratégias simples de comprar qualquer sinal positivo.
- `Prior`: prior sem views.
- `Markowitz`: referência original de média histórica.

Mesmos preços, 20 ações, horizontes escolhidos em 2024, janela de risco de 126 pregões, aversão 3, custos de 10 bps, piso/teto e calendário mensal. Sinal ao fechamento, execução no fechamento seguinte. Cardinalidades 1 a 20, mesmos 551 subconjuntos comuns e seleção própria por tamanho. São 3.997 trajetórias nas sete configurações, incluindo casos coincidentes; não são amostras independentes.

## Previsões necessárias

Os CSVs da avaliação preditiva excluíam previsões cujos resultados terminavam além de 2025. Esses cortes são adequados para calcular erros de teste, mas insuficientes para a simulação: o sinal de dezembro precisa existir mesmo sem ter desfecho conhecido dentro do ano, e erros maturados no início de 2025 podem depender de previsões feitas no final de 2024.

O exportador reconstrói o mesmo pool em todas as datas necessárias, usando os mesmos folds e a mesma purga. Não preenche previsões arbitrariamente. Reproduz numericamente as previsões já avaliadas. O motor usa resultados realizados apenas depois da data de maturidade. A seleção do grupo de features ocorreu com dados de 2024; a escolha de destacar o pool foi posterior ao diagnóstico de 2025 e continua exploratória.

## Resultados e auditoria

- `RELATORIO.txt`: interpretação da comparação.
- `comparacao_5_10_20.csv`: seleção própria em três tamanhos ilustrativos.
- `metricas.csv`: resultados das 3.997 simulações.
- `comparacao_pareada.csv` / `comparacao_resumo.csv`: diferenças nos mesmos pares e frequência descritiva de vitória.
- `series_selecao_propria.csv` e `rebalanceamentos_selecao_propria.csv`: trajetórias e pesos.
- `auditoria_comparada.csv`: previsão, Q, Omega e posterior em cada decisão.
- `auditoria_treino.csv` / `reproducao_previsoes.csv`: causalidade e reprodução da previsão conjunta.
- `verificacao.json`: exige reprodução numérica dos 1.142 resultados originais de BL e Markowitz e preservação dos arquivos de referência.
- `verificacao_adicional.json`: valida métricas finitas, casos únicos, pesos e limites, causalidade temporal e igualdade dos cenários com vinte ativos.

As séries dos subconjuntos comuns podem ser reconstruídas com o código e a semente; são guardados seus resultados e composições, evitando duplicar milhões de linhas de curvas.

## Reprodução

```powershell
python -m intel.model_training.exportar_pool_simulacao
python -m unittest intel.Blacklitterman.test_simulacao_pool intel.Blacklitterman.test_dois_cenarios
python -m intel.Blacklitterman.simular_pool_vs_atual
python -m analise_resultados.resumir_simulacao_pool
python -m analise_resultados.verificar_simulacao_pool
```

O exportador usa as mesmas dependências e versões dos testes de metodologia (numpy, pandas, XGBoost, scikit-learn, optuna importado pelo treinador e matplotlib para as figuras). O motor financeiro exige numpy e pandas. Não há novos downloads.

Uma melhora de RMSE não garante melhora da carteira. O gate de três notícias habilita seis dos 240 pares mensais, e a atualização de views pode atingir outros ativos pela covariância. Os resultados não são confirmação em período novo nem justificam escolher retrospectivamente a melhor cardinalidade. O estudo avalia a substituição do previsor sob o restante da configuração vigente, sem nova otimização de gate, alpha, risco ou custos.
