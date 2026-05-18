# Morpheus

> Хватит запускать AI-агентов с нуля.
>
> Morpheus генерирует `WAKE.md` - скомпилированный файл состояния проекта,
> который позволяет любому агенту продолжить работу с текущего места.

`README.md` объясняет проект людям.
`AGENTS.md` говорит агентам, как работать.
`WAKE.md` говорит агентам, где проект находится сейчас.

[English version](https://github.com/botAGI/morpheus-ai/blob/main/README.md)

> Статус: alpha. Последний опубликованный package release: v0.1.1.
> Deterministic compiler, receipts, CLI, API, UI launchpad, MCP endpoint,
> A2A-style discovery и cache-backed интеграции уже usable. В main есть
> review-gated v0.2 semantic alpha. LoRA/training экспериментален и не является
> core launch path.

![Morpheus terminal demo](https://raw.githubusercontent.com/botAGI/morpheus-ai/main/demo/morpheus-demo.gif)

## Зачем

Каждый AI-агент стартует холодным.

Вы снова вставляете контекст. Снова объясняете решения. Агент предлагает старые
идеи. Он не понимает, что изменилось вчера.

Morpheus компилирует текущее состояние проекта в `WAKE.md`, подкрепляя его
источниками, evidence и подписанными receipts.

```text
sources -> compile -> WAKE.md -> signed receipt -> agent handoff -> verify
```

## Главный Примитив

Morpheus - это Agent State Compiler.

Он генерирует `WAKE.md`: файл состояния проекта, который говорит агентам, где
проект сейчас, и связывает это состояние с источниками, evidence и подписанными
receipts.

Этот репозиторий намеренно коммитит
[WAKE.md](https://github.com/botAGI/morpheus-ai/blob/main/WAKE.md) как
публичный пример. Приватные проекты могут хранить `WAKE.md` внутри
`.morpheus/`.

## Быстрый Старт

Установка:

```bash
uvx --from morpheus-wake morpheus wake .
```

Через pipx:

```bash
pipx run --spec morpheus-wake morpheus wake .
```

Для приватных рабочих папок:

```bash
uvx --from morpheus-wake morpheus wake . --private
```

Это оставит скомпилированное состояние в `.morpheus/WAKE.md`.

Development install:

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -e ".[dev]"

morpheus wake .
```

## До / После

Без Morpheus:

```text
User: What changed yesterday?
Agent: I do not have enough context.
```

С Morpheus:

```text
User: Read WAKE.md. What changed yesterday?
Agent: Morpheus moved from "memory compiler" to "Agent State Compiler".
       Outdated: LoRA as the core product path.
       Current: WAKE.md with provenance receipts.
       Next action: review semantic candidates and expand richer stale-claim detection.
```

## Почему Не Просто Memory?

Memory говорит агенту, что происходило.
State говорит агенту, что правда сейчас.

RAG достаёт старые фрагменты.
Morpheus компилирует текущее состояние проекта.

`README.md` - для людей.
`AGENTS.md` - для инструкций агентам.
`WAKE.md` - для непрерывности агентной работы.

## Основные Возможности

- **WAKE.md compiler**: сканирует watched paths и извлекает решения, задачи,
  заметки, исправления и evidence.
- **Проверяемое происхождение**: пишет `state.json`, `evidence.jsonl` и
  подписанные ed25519 receipts с SHA-256 хешами.
- **Agent handoff**: создаёт инструкции, diagnostics и manifest URLs для другого
  coding agent.
- **Stale claim scan**: `morpheus stale .` находит старое позиционирование,
  конфликтующее с текущей рамкой `WAKE.md`.
- **Локальный UI launchpad**: setup, context sources, diagnostics, integrations,
  model smoke tests и handoff bundles.
- **Agent interop**: native `/agent/connect`, A2A-compatible Agent Card и
  минимальный MCP Streamable HTTP endpoint.
- **Context sources**: можно компилировать проект, monorepo, workspace или vault
  с заметками.
- **Integration cache readers**: GitHub, Gmail, Calendar, Slack и Linear могут
  добавлять evidence из локальных cache или token-backed adapters.

## Deterministic Core, Semantic Alpha

v0.1 намеренно deterministic. Он извлекает явные маркеры:

```text
TODO: DECISION: FIXME: NOTE: HACK: XXX:
```

Это делает receipts воспроизводимыми и простыми для проверки.

В main есть alpha semantic review path:

```bash
morpheus wake . --semantic --review
morpheus review list
morpheus review accept <candidate-id>
morpheus review apply
```

Semantic extraction остаётся review-gated. Candidates помечаются как
`source_backed` или `needs_review`, source spans проверяются перед apply, а
accepted claims становятся active только после `morpheus review apply` и нового
подписанного receipt.

## Obsidian И Личная База

Obsidian vault можно использовать как context source, потому что это папка с
Markdown-файлами. Правильный путь - сначала локальная компиляция: source links,
evidence, receipts и review. Не обучайте модель напрямую на сыром приватном
vault.

```bash
cd ~/Obsidian
morpheus wake . --private
```

Для workspace с несколькими проектами или vault выберите родительскую папку как
project root и настройте `.morpheus/morpheus.toml`:

```toml
watch_dirs = ["project-a", "project-b", "vault"]
```

## Agent Self-Connect

Агент может подключиться к Morpheus без чтения README:

```bash
morpheus prepare-agent
morpheus agent-connect --json
morpheus diagnostics --json
morpheus handoff
```

Когда HTTP API запущен:

```bash
curl -s "http://127.0.0.1:8000/agent/connect?project_root=$PWD"
curl -s "http://127.0.0.1:8000/agent/handoff.md?project_root=$PWD"
curl -s http://127.0.0.1:8000/.well-known/morpheus.json
curl -s http://127.0.0.1:8000/.well-known/agent-card.json
curl -s -X POST http://127.0.0.1:8000/mcp \
  -H 'Content-Type: application/json' \
  -H 'Accept: application/json, text/event-stream' \
  -d '{"jsonrpc":"2.0","id":1,"method":"tools/list"}'
```

Новый агент должен:

1. Прочитать `WAKE.md`.
2. Получить `/agent/connect` или выполнить `morpheus agent-connect --json`.
3. Следовать `next_action`.
4. Выполнить `morpheus compile` и `morpheus verify --all` после значимых правок.

## UI

```bash
morpheus serve --ui --host 127.0.0.1 --port 8000 --ui-port 5173
```

Открыть:

```text
http://127.0.0.1:5173/ui/index.html
```

Start screen позволяет выбрать project root, настроить watched paths, запустить
diagnostics, подготовить агента, проверить integrations, протестировать MCP tools
и скопировать полный handoff bundle.

## Архитектура

```text
morpheus/
  core/          compiler, models, receipts, verification, safe IO
  integrations/  filesystem and cache-backed external sources
  api/           FastAPI, agent connect, diagnostics, MCP, A2A card
  training/      experimental consolidation and LoRA helpers
ui/              static browser UI and Tauri shell
tests/           pytest suite for compiler, API, CLI, integrations, training
docs/            launch notes, testing notes, and product framing
```

Compile flow:

```text
morpheus compile
  -> scans configured watch_dirs
  -> extracts explicit evidence markers
  -> writes state.json and evidence.jsonl
  -> generates WAKE.md
  -> signs a receipt with ed25519
  -> links the receipt to the previous receipt hash
```

## CLI

| Command | Description |
| --- | --- |
| `morpheus wake .` | Init при необходимости, compile, verify и root `WAKE.md` |
| `morpheus wake . --private` | Compile и verify, но `WAKE.md` остаётся в `.morpheus/` |
| `morpheus stale .` | Найти устаревшие claims в публичном позиционировании |
| `morpheus init` | Инициализировать `.morpheus/` с config и keys |
| `morpheus compile` | Скомпилировать sources в `WAKE.md` и signed receipt |
| `morpheus verify --all` | Проверить receipt chain, signatures и latest artifacts |
| `morpheus wake` | Напечатать приватный `.morpheus/WAKE.md` |
| `morpheus prepare-agent` | Подготовить handoff для агента |
| `morpheus agent-connect --json` | Напечатать machine-readable agent manifest |
| `morpheus serve --ui` | Запустить FastAPI backend и browser UI |

## Разработка

```bash
make install-dev
make verify
make build
```

Полный quality gate описан в [docs/TESTING.md](docs/TESTING.md).

## Безопасность

Morpheus local-first. Не коммитьте `.morpheus/`, generated receipts, integration
caches, model outputs и token files. Используйте `127.0.0.1`, если это не trusted
LAN или authenticated proxy.

Подробнее: [SECURITY.md](SECURITY.md).

## Experimental Training

Local adapter training находится в `morpheus/training/`. Это optional,
explicit слой после reviewed state. Default path: compile, retrieve, cite
evidence, verify receipts.

## Лицензия

MIT
