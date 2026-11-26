# Avaliação das 300 notícias

Abra `AVALIAR_NOTICIAS.html` com duplo clique no Chrome ou Edge. Não exige instalação ou servidor. O arquivo já contém as 300 notícias da amostra; não contém o gabarito do modelo. Não há requisições externas.

1. Leia a empresa e a manchete.
2. Clique em Positivo, Negativo ou Neutro, ou use 1, 2 e 3. A tela avança automaticamente.
3. Use Anterior para corrigir. Observações são opcionais e salvas ao digitar.
4. O progresso fica no navegador. Para continuar, abra o mesmo arquivo no mesmo perfil. Guarde backups durante a avaliação.
5. Ao terminar, clique em **Baixar CSV preenchido** e entregue o arquivo para a análise. O nome é `rotulacao_humana.csv`; o navegador pode acrescentar um número se ele já existir na pasta de downloads.

Opcionalmente, use **Vincular CSV para salvar**, selecionando o CSV original em `analise_resultados/camadas/rotulacao_humana.csv`. O recurso depende de suporte do navegador. O arquivo escolhido é validado e há confirmação antes da primeira escrita. O vínculo precisa ser refeito ao reabrir a tela. Se não houver suporte ou permissão, o download funciona de forma independente.

**Backup do progresso** baixa JSON. **Importar CSV / backup** recupera respostas após trocar de navegador ou computador. A importação valida a amostra e pede confirmação antes de substituir o progresso.

Esta tela foi configurada para **um único avaliador**. A escolha preenche avaliador 1 e consenso, permitindo usar o avaliador existente no repositório. O segundo avaliador fica vazio. Não tratar isso como consenso entre dois avaliadores no TCC. A cópia original do CSV não foi preenchida automaticamente pelo desenvolvimento.

Os critérios estão disponíveis em “Critérios e salvamento”. Avalie a mensagem sobre a empresa, sem consultar o retorno posterior ou o gabarito. Uma notícia sem direção clara pode ser neutra; registre a ambiguidade na observação.

## Desenvolvimento e verificações

`index.html`, `app.js` e `noticias.js` são as fontes. Para gerar novamente o HTML único: `python analise_resultados/avaliacao_noticias/empacotar.py`.

Teste de lógica, com Node: `node analise_resultados/avaliacao_noticias/test_app.cjs`. Verifica classificação, correção, persistência, conclusão e exportação/importação de CSV com aspas e quebras de linha. Não substitui QA visual. A abertura para inspeção no navegador interno foi bloqueada pela política de protocolos locais; a gravação via seletor nativo também não foi testada visualmente.
