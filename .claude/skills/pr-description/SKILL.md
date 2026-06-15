---
name: pr-description
description: Gera título e descrição em Markdown para Pull Requests deste projeto, no formato Conventional PR (Summary, Changes, Test plan, Related). Use ao abrir um PR ou quando o usuário pedir a descrição/mensagem do PR.
---

# Descrição de Pull Request

Gera o título e o corpo (Markdown) de um PR para o `ecommerce-recsys-mlops`, baseado nos
commits da branch atual em relação a `main`.

## Passo 1 — Coletar contexto

```bash
git branch --show-current
git log main..HEAD --oneline
git diff main...HEAD --stat
```

## Passo 2 — Identificar o ID da tarefa (opcional)

Se o nome da branch seguir o padrão `<tipo>/<ID>-<slug>` (ex:
`feat/TCF1-52-churn-dataset-dataloader`), extraia o `<ID>` (ex: `TCF1-52`).

Se a branch não tiver esse padrão (ex: `feat/ajusta-readme`), **não há ID** — siga sem a
seção `## Related` e sem prefixo no título.

## Passo 3 — Montar o título

- Com ID: `[<ID>] <descrição curta no imperativo>`
- Sem ID: `<descrição curta no imperativo>`

A descrição curta resume o objetivo geral da branch (não é só o último commit).

## Passo 4 — Montar a descrição (template)

```markdown
## Summary
- <1-3 bullets descrevendo o que foi feito e por quê>

## Changes
- <bullets das mudanças principais — módulos/arquivos afetados, decisões relevantes>

## Test plan
- [ ] <como validar — ex: `make check`, testes específicos, execução manual>

## Related
- <ID da tarefa, ex: TCF1-52>
```

Regras:
- `## Related` só aparece se houver ID de tarefa identificado no Passo 2. Se não houver,
  omita a seção inteira.
- `Summary` foca no "porquê"/objetivo; `Changes` foca no "o quê" (arquivos, módulos,
  decisões técnicas). Não repita a lista de commits literalmente — sintetize.
- `Test plan` deve ser um checklist verificável (comandos do Makefile deste projeto:
  `make check`, `make test`, `make lint`, etc., ou passos manuais quando aplicável).
- Linguagem: português, seguindo a convenção de commits do projeto.
