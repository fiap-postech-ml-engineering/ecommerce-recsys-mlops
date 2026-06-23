---
name: commit
description: Regras de commit e branch deste projeto — formato de mensagem, quando commitar, push e proteção de branch. Use sempre que for criar um commit ou abrir uma branch neste repositório.
---

# Commits e branches

## Formato da mensagem

`<tipo>(<escopo>): <descrição no imperativo, em português>`

Tipos: `feat`/`fix`/`refactor`/`test`/`docs`/`chore`. Commits pequenos, um por
responsabilidade lógica — não acumular mudanças não relacionadas num único commit.

## Branches

`<tipo>/<slug-descritivo>`, com `-<ID-da-tarefa>` opcional quando houver card no Kanban
(ex: `feat/TCF1-52-churn-dataset-dataloader`). Veja a skill `notion-kanban` para como obter
o ID da tarefa atual.

## Quando commitar

Fluxo: implementar → usuário valida → só rodar `make check` e criar o commit quando
solicitado explicitamente ("commita", "abre PR", etc). Nunca commitar proativamente sem
pedido explícito.

Se um novo pedido de implementação chegar antes do ciclo anterior ser fechado (check +
commit), avisar que o escopo mudou e sugerir finalizar o ciclo atual antes de seguir.

## Regras do commit em si

- **Não incluir** a linha `Co-Authored-By: Claude ...` (ou qualquer menção ao Claude Code)
  no corpo do commit.
- **Sempre fazer `git push`** depois de criar um commit.
- **Commits sempre em branch própria**, nunca direto em `main`. Se `git push` indicar que
  uma regra de proteção de branch foi *bypassada* (ex: "Bypassed rule violations" /
  "Changes must be made through a pull request"), o push foi para `main` por engano — parar
  imediatamente e perguntar ao usuário se quer continuar assim ou desfazer e refazer em uma
  branch com PR.

## Quando usar `grill-me` antes de comitar

Se o card do Kanban (via `notion-kanban`) for complexo ou ambíguo, usar a skill `grill-me`
para alinhar o entendimento do escopo com o usuário **antes** de implementar — evita
retrabalho e commits que precisam ser refeitos por mal-entendido de requisito.
