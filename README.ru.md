# Morpheus AI

**Локальный компилятор памяти для AI-агентов с проверяемым происхождением.**

Morpheus превращает файлы проекта, заметки, решения, задачи и опциональные
экспорты интеграций в компактный handoff (`WAKE.md`), которому могут доверять
люди и агенты. Агент не стартует с нуля: он получает актуальное состояние,
доказательства и цепочку receipt-файлов, показывающую источник этого состояния.

[English version](README.md)

> Статус: alpha. Компилятор, receipts, CLI, API, UI launchpad, MCP endpoint и
> cache-backed интеграции уже usable. Scheduled LoRA training пока
> экспериментальный слой памяти, а не основной продуктовый путь.

## Зачем Нужен Morpheus

Агенты теряют контекст между сессиями. RAG помогает находить текст, но часто не
отвечает на ключевые вопросы: какие факты актуальны, откуда они взялись и что
следующий агент должен сделать первым.

Morpheus строится вокруг более строгого цикла:

```text
sources -> compile -> WAKE.md -> signed receipt -> agent handoff -> verify
```

Это полезно для:

- передачи проекта от одного агента другому,
- компиляции Obsidian vault или рабочей папки в agent-readable memory,
- привязки решений, задач и заметок к исходным файлам,
- локального доступа через CLI, HTTP, A2A-style discovery и MCP tools,
- проверки готовности локальной модели и integration cache.

## Основные Возможности

- **WAKE.md compiler**: сканирует watched paths и извлекает решения, задачи,
  заметки, исправления и evidence-маркеры.
- **Проверяемое происхождение**: пишет `state.json`, `evidence.jsonl` и
  подписанные ed25519 receipts с SHA-256 хешами.
- **Agent handoff**: создаёт инструкции, diagnostics и manifest URLs для другого
  coding agent.
- **Локальный UI launchpad**: setup, context sources, diagnostics, integrations,
  model smoke tests и handoff bundles.
- **Agent interop**: native `/agent/connect`, A2A-compatible Agent Card и
  минимальный MCP Streamable HTTP endpoint.
- **Context sources**: можно компилировать проект, monorepo, workspace или vault
  с заметками.
- **Integration cache readers**: GitHub, Gmail, Calendar, Slack и Linear могут
  добавлять evidence из локальных cache или token-backed adapters.
- **Экспериментальный training pipeline**: consolidation в JSONL и LoRA adapters
  только при явном включении.

## Установка

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -e ".[dev]"
```

## Быстрый Старт

```bash
# Инициализировать Morpheus state в текущем проекте.
morpheus init

# Скомпилировать watched sources в WAKE.md, state.json, evidence.jsonl и receipt.
morpheus compile

# Проверить receipt chain и последние artifacts.
morpheus verify --all

# Вывести текущую compiled memory.
morpheus wake
```

## UI

Запустить backend и static browser UI:

```bash
morpheus serve --ui --host 127.0.0.1 --port 8000 --ui-port 5173
```

Открыть:

```text
http://127.0.0.1:5173/ui/index.html
```

Для trusted LAN:

```bash
morpheus serve --ui --host 0.0.0.0 --port 8000 --ui-port 5173
```

Start screen позволяет выбрать project root, настроить watched paths, запустить
diagnostics, подготовить агента, проверить integrations, протестировать MCP tools
и скопировать полный handoff bundle.

## Obsidian И Личная База

Obsidian vault можно использовать как context source, потому что это папка с
Markdown-файлами. Правильный путь не “скормить всю базу в веса”, а сначала
локально компилировать и извлекать память с source links. Только стабильные и
проверенные воспоминания стоит продвигать в будущий training dataset.

Пример:

```bash
cd ~/Obsidian
morpheus init
morpheus compile
morpheus verify --all
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

1. Получить `/agent/connect` или выполнить `morpheus agent-connect --json`.
2. Следовать `next_action`.
3. Прочитать `WAKE.md` перед изменениями.
4. Выполнить `morpheus compile` и `morpheus verify --all` после значимых правок.

## Интеграции

Показать adapters:

```bash
morpheus integrate --list
morpheus integrate --list --json
```

Текущие adapters:

- `github`: issues, pull requests, commits и cached metadata.
- `gmail`: local Gmail cache и OAuth-oriented token path.
- `calendar`: local Calendar cache и OAuth-oriented token path.
- `slack`: local Slack export cache плюс optional token file.
- `linear`: local Linear issue cache плюс optional token file.

Локальные tokens и caches по умолчанию живут вне репозитория: `~/.morpheus/`.

## Архитектура

```text
morpheus/
  core/          compiler, models, receipts, verification, safe IO
  integrations/  filesystem and cache-backed external sources
  api/           FastAPI, agent connect, diagnostics, MCP, A2A card
  training/      experimental consolidation and LoRA training helpers
ui/              static browser UI and Tauri shell
tests/           pytest suite for compiler, API, CLI, integrations, training
docs/            release and testing notes
```

Compile flow:

```text
morpheus compile
  -> scans configured watch_dirs
  -> extracts markers such as TODO:, DECISION:, FIXME:, NOTE:, HACK:, XXX:
  -> writes state.json and evidence.jsonl
  -> generates WAKE.md
  -> signs a receipt with ed25519
  -> links the receipt to the previous receipt hash
```

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

## Лицензия

MIT
