# Morpheus

> Хватит позволять coding agents галлюцинировать о вашем репозитории.
>
> First verify. Then learn.

Morpheus проверяет утверждения агентов о проекте по source-backed состоянию. А
затем может запустить автономную learning lab, чтобы проверить, можно ли
дистиллировать устойчивую проектную правду в локальные веса.

`README.md` объясняет проект людям.
`AGENTS.md` говорит агентам, как работать.
`WAKE.md` говорит агентам, где проект находится сейчас.

[English version](https://github.com/botAGI/morpheus-ai/blob/main/README.md)

> Статус: alpha. Последний опубликованный package release: v0.1.1.
> Deterministic compiler, receipts, CLI, API, UI launchpad, MCP endpoint,
> A2A-style discovery и cache-backed интеграции уже usable. В main есть
> review-gated v0.2 semantic/check work и экспериментальная autonomous learning
> lab. Local adapter learning экспериментален до прохождения eval; source spans
> остаются источником истины.

![Morpheus terminal demo](https://raw.githubusercontent.com/botAGI/morpheus-ai/main/demo/morpheus-demo.gif)

## Зачем

Каждый AI-агент стартует холодным.

Вы снова вставляете контекст. Снова объясняете решения. Агент предлагает старые
идеи и утверждает, что в проекте есть несуществующие возможности.

Morpheus компилирует проектное состояние, проверяет текст агента по
source-backed evidence и может собрать экспериментальный локальный learning
dataset только из accepted claims.

```text
sources -> WAKE.md -> morpheus check -> reviewed dataset -> local adapter lab
```

## Главный Примитив

Morpheus - это source-grounded truth layer с экспериментальным learning core.

Он генерирует `WAKE.md`: файл состояния проекта, который говорит агентам, где
проект сейчас. `morpheus check` проверяет claims по локальному состоянию, source
spans, manifests и evidence. `morpheus learn lab` запускает автономный локальный
эксперимент: можно ли превратить проверенную проектную правду в полезную
adapter memory.

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

Alpha loop из трёх команд:

```bash
uvx --from morpheus-wake morpheus wake .
gh pr view 42 --json body -q .body | morpheus check
morpheus learn lab . --backend mlx
```

`morpheus learn lab` экспериментален. Он может использовать strict autonomous
benchmark lane, но никогда не активирует adapters автоматически и не делает raw
Markdown fine-tuning.

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
User: Check this agent answer before I merge it.
Agent: stale: "Morpheus is mainly a LoRA trainer."
       incorrect: "morpheus check sends text to cloud by default."
       verified: "The package name is morpheus-wake."
```

## Почему Не Просто Memory?

Memory говорит агенту, что происходило.
Source-grounded state говорит агенту, что сейчас подтверждено источниками.

RAG достаёт старые фрагменты.
Morpheus проверяет текущие project claims перед любым learning experiment.

`README.md` - для людей.
`AGENTS.md` - для инструкций агентам.
`WAKE.md` - для непрерывности агентной работы.

## Основные Возможности

- **WAKE.md compiler**: сканирует watched paths и извлекает решения, задачи,
  заметки, исправления и evidence.
- **Local claim check**: `morpheus check` проверяет текст агента из файла или
  stdin по локальному состоянию и возвращает `verified`, `stale`, `incorrect`
  или `unknown`.
- **Autonomous learning lab**: `morpheus learn lab` строит strict benchmark
  dataset из machine-verifiable source-backed claims, при необходимости запускает
  local MLX LoRA smoke training и пишет pass/partial/fail report без активации
  adapters.
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

## Deterministic Core, Check, And Learning Alpha

v0.1 намеренно deterministic. Он извлекает явные маркеры:

```text
TODO: DECISION: FIXME: NOTE: HACK: XXX:
```

Это делает receipts воспроизводимыми и простыми для проверки.

`morpheus check` в v0.2 alpha slice по умолчанию работает только локально. Он
не отправляет текст агента или source excerpts в cloud providers.

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

Learning core находится за этим gate:

```bash
morpheus learn dataset . --from accepted
morpheus learn train . --dry-run
morpheus learn eval .
morpheus learn lab . --no-train
```

Нет accepted source span - нет training example. Нет успешного eval - нет
activation. Нет rollback - нет production use.

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
| `morpheus check` | Проверить текст агента из stdin по локальному состоянию проекта |
| `morpheus check --input FILE` | Проверить текст агента из файла |
| `morpheus check --json` | Напечатать machine-readable check result |
| `morpheus review list` | Показать semantic candidates для review |
| `morpheus review apply` | Применить accepted candidates в active state и подписать receipt |
| `morpheus learn lab .` | Запустить autonomous learning lab без activation adapters |
| `morpheus learn dataset .` | Собрать dataset из accepted source-backed candidates |
| `morpheus learn status` | Показать learning dataset и adapter status |
| `morpheus learn train . --dry-run` | Сгенерировать training artifacts без обучения |
| `morpheus learn eval .` | Запустить eval harness для latest dataset или planned adapter |
| `morpheus stale .` | Найти устаревшие claims в публичном позиционировании |
| `morpheus init` | Инициализировать `.morpheus/` с config и keys |
| `morpheus compile` | Скомпилировать sources в `WAKE.md` и signed receipt |
| `morpheus verify --all` | Проверить receipt chain, signatures и latest artifacts |
| `morpheus wake` | Напечатать приватный `.morpheus/WAKE.md` |
| `morpheus prepare-agent` | Подготовить handoff для агента |
| `morpheus agent-connect --json` | Напечатать machine-readable agent manifest |
| `morpheus serve --ui` | Запустить FastAPI backend и browser UI |

Default semantic alpha provider - локальный/offline heuristic extraction. Он
никогда не вызывает cloud providers, если пользователь явно не настроил
provider.

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
