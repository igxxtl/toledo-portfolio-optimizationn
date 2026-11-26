# Qualidade preditiva das camadas

Reprodução: `python -m analise_resultados.avaliar_camadas_preditivas`. Exige numpy, pandas e matplotlib. O código reproduz a auditoria híbrida antes de calcular controles e atualiza a seção correspondente no mesmo TeX aberto. Não treina, não refaz tuning e não altera os resultados financeiros.

## Perguntas respondidas

1. XGBoost e Q: a mistura melhora a previsão? Comparar XGB com Q.
2. A polaridade textual acrescenta valor? Comparar Q_neutro com Q e BL_neutro com BL.
3. A atualização BL melhora a previsão quantitativa? Comparar XGB com BL_XGB.
4. A cadeia completa supera as partes? Comparar BL com XGB, Q e prior.

`previsoes_e_resultados.csv` guarda cada previsão e resultado. `scores.csv` traz N, MAE, RMSE, MSE, HR, acurácia balanceada, R², Pearson, viés, dispersão e skill contra excesso zero, para toda a amostra, por ativo e nos subconjuntos com notícias, sem notícias e gate ativo. `ranking_horizonte.csv` ordena RMSE e HR separadamente. `efeito_pareado.csv` mede reduções de erro e mudanças de acurácia entre os controles nas mesmas observações.

## Controle e alvos

São 240 pares ativo-data nas decisões mensais de 2025. Para o horizonte selecionado antes do teste, 228 têm desfecho até 30/12/2025; 12 são excluídos e documentados. Existem 40 pares com notícias e 6 com gate ativo.

O controle neutro zera somente o componente de sentimento em Q. A cobertura, os pesos quantitativos dependentes da contagem, Omega, gate, prior e covariância não mudam. BL_XGB substitui Q pela previsão XGB diária e preserva esses mesmos controles. Nenhuma das variantes representa retirar todas as informações de notícias do sistema, pois sua quantidade ainda influencia a confiança.

Alvo principal: log-retorno do horizonte dividido por h, menos a referência diária livre de risco; compara o XGB em escala diária equivalente e o BL diário. A compatibilidade com a escala aritmética do prior/BL é aproximada. HR é do excesso de retorno, diferente de HR bruto. Para sensibilidade, há avaliação no retorno simples do pregão seguinte com 240 pares: as views em log são convertidas com expm1, mas isso não transforma o modelo multiperíodo em um modelo treinado para um dia. São duas perguntas preditivas diferentes.

O score `skill_zero = 1 - MSE_modelo/MSE_zero` usa a referência de excesso zero (taxa livre de risco). Positivo significa melhora contra ela. Não é probabilidade nem nota limitada entre zero e cem. RMSE menor, HR/BA/R²/skill maiores são melhores; não há um vencedor universal quando esses critérios divergem.

Os ganhos são descritivos desta rodada, com escolhas retrospectivas e dependência entre observações. Não há alegação de significância ou generalização. O gráfico foi inspecionado e o projeto LaTeX compilou com sucesso pelo editor após ajustes das dependências de TikZ e da extensão explícita do input.
