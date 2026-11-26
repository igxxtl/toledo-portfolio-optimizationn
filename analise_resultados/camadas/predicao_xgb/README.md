# Diagnóstico de predição XGBoost

Avaliação dos artefatos existentes em 2025, com horizonte escolhido em 2024 e desfechos conhecidos até 30/12/2025. Não houve retreinamento ou substituição do horizonte com base no teste. Foram avaliados 20 ativos × 3 horizontes (5, 10 e 21 pregões), com métricas anuais e trimestrais. O recorte dos horizontes selecionados contém 4.683 pares, sem independência entre alvos sobrepostos.

Reproduzir na raiz do repositório, com numpy, pandas e matplotlib:

```powershell
python -m analise_resultados.avaliar_predicao_xgb
python -m unittest analise_resultados.test_predicao_xgb
```

O comando de diagnóstico também copia o fragmento e as três figuras para `Documents/TCC 2/05_Resultados` e insere sua chamada antes da comparação financeira com ativos comuns. O script de integração das camadas deixa de incluir a avaliação humana de sentimento na discussão principal; seus dados permanecem preservados.

## Métricas e arquivos

- `metricas.csv`: MAE, RMSE, R², acurácia direcional, acurácia balanceada, viés, dispersão e comparações por ativo/horizonte/trimestre. `selected` marca a escolha anterior ao teste.
- `observacoes.csv`: previsão e referências em cada data; `maturity` é a data em que o alvo termina.
- `manifest.json`: resumo, critérios, limitações e hashes das entradas.
- `trimestres_selecionados.csv`: médias simples entre ativos dos horizontes selecionados. O trimestre é o de origem da previsão.
- `horizontes_diagnostico.csv`: comparação descritiva dos três horizontes; não seleciona novamente o modelo.
- `shap_historico.csv` e `shap_relativo_medio.csv`: atributos dos modelos salvos. Cada ativo tem seu SHAP absoluto normalizado pela soma de seus atributos, antes da média entre os vinte ativos. Atributos ausentes contribuem com zero. Esses resumos não são SHAP exclusivo do teste ou de cada previsão OOF.

As referências expansivas usam somente alvos já maturados no fechamento da previsão e exigem 80 observações: média dos log-retornos de mesmo horizonte e direção mais frequente de seus sinais (-1, 0, +1). Em empate, usa-se essa ordem. A referência de direção não tem magnitude predita nem RMSE. Retorno zero é apenas referência de magnitude, não um previsor binário de alta/baixa.

`skill_zero = 1 - MSE_XGB/MSE_zero`; `skill_historical` usa o MSE da média histórica. Skill positivo indica melhora do erro. R² usa a média retrospectiva dos alvos do teste como referência, portanto pode ser negativo mesmo com skill contra zero positivo. Não comparar RMSE bruto de horizontes distintos como se fossem o mesmo alvo.

Não foram refeitas as escolhas de estimador, hiperparâmetros ou folds dos artefatos originais. Assim, selecionar horizonte em 2024 não torna toda a cadeia independente de 2025. As comparações são diagnósticas; não há testes iid ou alegação de significância/generalização. A revisão visual dos PNGs foi realizada; a compilação e paginação LaTeX devem ser conferidas no Overleaf.
