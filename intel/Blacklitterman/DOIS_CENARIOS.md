# Comparação por número de ativos: dois cenários

Experimento **exploratório** em 2025, n = 1…20, BL híbrido versus Markowitz versus uma aplicação hipotética rendendo Selic constante de 15% a.a. A rodada atual usa **gate de 3 notícias**, solicitado após a sensibilidade de 1 a 5 em 2025. A escolha do limiar é retrospectiva e não foi validada num teste independente. Saídas em `outputs/dois_cenarios/`; a rodada anterior de gate 1 foi preservada em `outputs/dois_cenarios_gate1_historico/`. A primeira execução com gate 5 permanece identificada como histórica em `outputs/dois_cenarios_gate5_historico/`.

## Execução

Na raiz do repositório, com Python, numpy e pandas:

```powershell
python -m intel.Blacklitterman.simular_dois_cenarios --repeticoes 30
python -m unittest intel.Blacklitterman.test_dois_cenarios
python -m intel.Blacklitterman.verificar_dois_cenarios
```

Não há downloads, chamadas a modelos ou consulta de preços externos. A venv antiga do checkout pode apontar para um Python ausente; selecione um interpretador funcional.

## Cenários

**Mesmos ativos:** subconjunto fixo durante o ano, recebido pelos dois métodos, com pesos recalculados mensalmente. Até 30 subconjuntos distintos por n, sorteados sem reposição com semente 2025+n. Se existirem até 30 combinações, todas são usadas. Assim, n=1 e n=19 usam 20 conjuntos; n=20 usa um. Não é uma enumeração de todas as combinações intermediárias. A mediana e P10/P90 descrevem a dispersão dos resultados; não são intervalos de confiança nem uma carteira implementável composta pelas medianas.

**Escolha livre:** a cada rebalanceamento, cada estratégia ranqueia o universo pelo retorno diário excedente esperado dividido pela volatilidade histórica. Markowitz usa a média histórica; BL usa seu retorno posterior híbrido. Os n primeiros entram na carteira. Trata-se de uma regra explícita adicionada aos modelos, e não de uma seleção universalmente ótima de cardinalidade n. Não há busca combinatória do melhor conjunto pelo resultado futuro.

Nos dois cenários a alocação maximiza `mu_excesso'w - (3/2) w'Cov w`, com soma dos pesos = 1 e posições compradas. A mesma função objetivo e a mesma aversão ao risco são aplicadas aos dois modelos. O otimizador resolve o problema com restrições diretamente, sem simplesmente cortar pesos negativos de uma solução irrestrita.

Para que n seja o número efetivo de posições, o piso de peso é 0,1% para cada selecionada. O teto é `max(MAX_WEIGHT_PER_ASSET, 1/n)`: n=1 tem teto 100%, n=2 tem teto 50%. Em n=2, os pesos-alvo são necessariamente 50/50; diferenças no cenário livre decorrem da seleção. Teto e piso são aplicados nas operações; pesos podem se afastar desses limites pela variação dos preços entre operações.

## Informação e execução

- Teste: datas comuns dos preços locais em 2025; os dias iniciais em caixa rendem a taxa diária da Selic.
- Sinal: fechamento da primeira data disponível de cada mês; execução: fechamento do pregão seguinte. A nova posição começa a capturar retornos no pregão posterior à execução.
- Preços, covariância e médias: janela móvel de 126 observações terminando no fechamento do sinal. Usa preços de fechamento locais e converte para retornos simples.
- Horizonte XGB: escolha entre 5, 10 e 21 dias pela acurácia direcional na validação de 2024, incluindo apenas desfechos integralmente conhecidos antes de 2025. Empates favorecem o horizonte menor. Os arquivos por horizonte são usados diretamente.
- Confiança e escala do sentimento: apenas previsões antigas cujo horizonte completo já terminou na data do sinal. Nunca se usa o erro futuro da previsão corrente. A confiança do modelo é agregada sobre os últimos erros maduros, e não condicionada ao erro corrente.
- Notícias: usa exclusivamente o dia civil anterior ao sinal. É uma convenção conservadora para evitar incorporar manchetes publicadas depois do fechamento; não representa uma janela acumulada de notícias de fim de semana.
- Gate de notícias: `NEWS_COUNT_MIN = 3`, com hard prior, alpha e tau mantidos. Há 6 pares ativo-data elegíveis em 240 observações mensais, em 3 das 12 datas de sinal. O gate foi escolhido após examinar a sensibilidade de 2025, portanto a avaliação é exploratória.
- Prior BL: equilíbrio com pesos iguais e aversão ao risco fixa de 3. Não usa market cap de 2026 para o teste histórico. O posterior é estimado no universo completo e depois restringido aos ativos selecionados. Portanto, o cenário comum compara os mesmos ativos elegíveis, permitindo que o BL utilize seu universo de informação para formar as estimativas.
- As views XGB são aproximadas em escala diária por divisão pelo horizonte; variância dos erros maduros em escala diária segue a convenção h² do projeto. O alinhamento entre log-retorno previsto e retorno simples é uma aproximação.
- Posições variam diariamente com os preços; só há operações nas datas de execução.
- Custo: 10 bps sobre meia L1 dos valores negociados, incluindo caixa; calculado de forma autofinanciada. Entrada inicial tem custo, liquidação final não é simulada. Este é um modelo simplificado de custo, não uma tabela de corretagem por ordem.
- Selic: `(1.15)^(1/252)-1`, nas mesmas datas das carteiras. O retorno do período pode ser inferior a 15%.
- Sharpe: média do excesso diário à Selic dividida pelo desvio-padrão desse excesso, anualizada por sqrt(252). Drawdown inclui o capital inicial antes da primeira observação.

## Arquivos

| Arquivo | Conteúdo |
|---|---|
| `resumo.csv` | Medianas e percentis por cenário, n e modelo |
| `metricas.csv` | Métricas de cada simulação individual |
| `comparacao_pareada.csv` | Retorno BL menos Markowitz, em pontos percentuais, para o mesmo conjunto |
| `series.csv` | Retornos, capitais, turnover e custos diários de todas as simulações |
| `rebalanceamentos.csv` | Datas de sinal/execução, ativos escolhidos e pesos-alvo |
| `subconjuntos.csv` | Composição dos conjuntos comuns |
| `horizontes_validacao.csv` | Horizonte escolhido e desempenho anterior ao teste |
| `auditoria_sinais.csv` | Datas de previsão, disponibilidade dos desfechos, notícias e Ω |
| `comparacao.svg` | Curvas de retorno por n, com dispersão dos subconjuntos |
| `manifest.json` | Parâmetros, limitações e hashes dos dados de entrada |
| `verificacao.json` | Verificações dos resultados materializados |

## Resultado da execução com 30 repetições

Foram concluídas 1.142 simulações e 285.500 observações diárias (250 datas por carteira). Cinco testes automatizados e as verificações dos artefatos passaram. Os valores abaixo são retornos líquidos acumulados, não anualizados.

| n | BL, mesmos ativos (mediana) | Markowitz, mesmos ativos (mediana) | BL, escolha livre | Markowitz, escolha livre |
|---:|---:|---:|---:|---:|
| 1 | 21,26% | 21,26% | −10,37% | 44,24% |
| 5 | 27,85% | 19,33% | 58,40% | 22,25% |
| 10 | 33,02% | 11,01% | 40,14% | 18,84% |
| 15 | 31,48% | 20,89% | 36,21% | 19,03% |
| 20 | 28,21% | 19,10% | 28,21% | 19,10% |

Selic no período: **14,87%**, equivalente a 15% a.a. nas mesmas 250 datas. Para n=1 a mediana usa as 20 ações individuais; n=20 possui apenas um conjunto. As linhas intermediárias acima usam 30 subconjuntos comuns cada. A escolha livre corresponde a uma estratégia dinâmica por modelo e n, não à mediana de sorteios. Consulte `resumo.csv` para todos os n e demais métricas; o desenho completo está em `comparacao.svg` (também exportado como `comparacao.png`). Os exemplos desta tabela não representam uma escolha de n feita antes do teste.

## Limites de interpretação

Este experimento muda o motor de execução, o prior, a estimativa de confiança e o otimizador em relação ao pipeline histórico; seus valores **não devem ser misturados** aos CSVs antigos. Ele reutiliza os sinais XGB existentes. A escolha original entre classificador e regressor pode ter considerado todo o período OOF, inclusive o teste. Não houve novo treinamento com validação temporal aninhada nem auditoria integral das features e dos folds. Por isso, corrigir a causalidade do motor não transforma o experimento em uma demonstração definitiva de superioridade fora da amostra.

Outras limitações: universo fixo de 20 ações, possível viés de sobrevivência, histórico de fechamento ajustado, ausência de spreads e preços reais de execução, hiperparâmetros herdados de pesquisas anteriores e apenas um ano de teste. Escolher o melhor n pelo resultado de 2025 é uma conclusão retrospectiva. Um n operacional deve ser escolhido em validação anterior e avaliado em outro período.

Os testes cobrem projeção com limites, solução analítica de alocação, execução ao fechamento, evolução dos pesos, custos, reprodutibilidade dos conjuntos e invariância do primeiro sinal a alterações em retornos e desfechos futuros.
