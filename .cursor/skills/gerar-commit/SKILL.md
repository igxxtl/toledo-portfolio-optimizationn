---
name: gerar-commit
description: Analisa os arquivos em staged e gera uma mensagem de commit semântica em Português do Brasil. Use quando o usuário pedir para commitar ou rodar o comando /gerar-commit.
---

# Agent Skill: Gerador de Commit Semântico

## Objetivo
Analisar as alterações em `staged` no repositório local e gerar uma mensagem de commit precisa, descritiva e baseada em boas práticas, escrita em Português do Brasil.

## O que analisar
- O `diff` dos arquivos atualmente em `staged`.
- O contexto da mudança (ex: adição de feature, correção de bug, refatoração, atualização de dependência).
- Impactos potenciais na arquitetura ou comportamento do sistema.

## Padrões de Commit
- **Conventional Commits:** Use os prefixos padrão em inglês (ex: `feat:`, `fix:`, `chore:`, `docs:`, `refactor:`, `test:`). Não traduza os prefixos.
- **Título (Subject):** Deve ser EXTREMAMENTE SUCINTO e direto ao ponto, com no máximo 50 caracteres. Comece a descrição com um verbo no imperativo. Exemplo: `feat: adiciona botão de login`.
- **Corpo (Body):** Pule uma linha em branco após o título. Explique o *porquê* da mudança e o contexto (o "como" já está visível no código). Quebre as linhas em até 72 caracteres. Se o commit for muito simples, omita o corpo.
- **Rodapé (Footer):** Se houver uma quebra de compatibilidade, adicione `BREAKING CHANGE:` seguido da explicação. Opcionalmente, cite números de tickets/issues.

## Guardrails
- Não invente contexto: Baseie-se estritamente no código que está em `staged`. Se o contexto for muito pequeno, mantenha o commit simples e sucinto.
- Evite mensagens genéricas: Nunca gere mensagens como `fix: corrige bug` ou `chore: atualiza arquivos`. Seja específico, mas breve.
- Apenas a mensagem: Não adicione saudações, notas extras ou explicações sobre como você pensou.

## Output
Em vez de apenas imprimir a mensagem, use sua ferramenta de terminal para executar o commit diretamente.
Crie o comando no formato `git commit -m "titulo sucinto" -m "corpo da mensagem e breaking changes se houver"`.
Se a mensagem não tiver corpo, use apenas `git commit -m "titulo sucinto"`.