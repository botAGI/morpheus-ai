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

> Статус: beta release. Latest GitHub release и beta package: v0.2.0b1.
> Deterministic compiler, local claim checker, receipts, CLI, API, UI launchpad,
> MCP truth tools, A2A-style discovery, cache-backed интеграции и autonomous
> learning lab уже usable. Local adapter learning экспериментален до
> прохождения eval; source spans остаются источником истины.
> Для v0.2-функций пиньте `morpheus-wake==0.2.0b1`; unpinned PyPI tools могут
> выбрать stable v0.1.1 вместо этой beta.
>
> Последний live dogfood stability gate на main: repeat-2 `ML_CORE_PASS` с 69
> strict source-backed candidates, 290 training examples, full base-vs-adapter
> eval coverage, zero critical failures и без автоматической adapter activation.
> См.
> [`docs/reports/ML_CORE_LIVE_REPORT.md`](docs/reports/ML_CORE_LIVE_REPORT.md).

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

## Roadmap

Morpheus не пытается стать ещё одним review bot. Следующая продуктовая ось -
verified classification-to-training pipeline:

- **v0.3**: semantic classifier для architecture, implementation, product,
  security, command, integration, stale, convention, task и temporary facts.
- **v0.4**: dataset quality dashboard для trainable, retrievable, stale, unsafe,
  needs-review, negative и eval-only claims.
- **v0.5 (завершён в текущем коде)**: canonical adapter memory benchmark с
  category-level base-vs-adapter deltas и activation/rollback gates.
- **v0.6 (реализован; дальше hardening подписанного authority)**: audited
  routing между prompt, retrieval, adapter training, eval, negative examples,
  stale archive и human review; persisted lifecycle transitions теперь проходят
  через единую границу канонического пересчёта. Осталось обеспечить явный review
  authority для подписанного compiled active state.
- **v0.7 (local core завершён; дальше orchestration)**: review-gated team
  feedback идемпотентен и никогда не активирует adapter автоматически; осталось
  объединить все документированные team signals в один ingestion path.

См. [docs/ROADMAP.md](docs/ROADMAP.md). Инвариант строгий: нет accepted source
span - нет training example, нет успешного eval - нет adapter activation, а
adapter output не является source of truth.

## Быстрый Старт

Установка v0.2 beta:

```bash
uvx --from 'morpheus-wake==0.2.0b1' morpheus wake .
```

Через pipx:

```bash
pipx run --spec 'morpheus-wake==0.2.0b1' morpheus wake .
```

Для приватных рабочих папок:

```bash
uvx --from 'morpheus-wake==0.2.0b1' morpheus wake . --private
```

Это оставит скомпилированное состояние в `.morpheus/WAKE.md`.

Alpha loop из трёх команд:

```bash
uvx --from 'morpheus-wake==0.2.0b1' morpheus wake .
gh pr view 42 --json body -q .body | uvx --from 'morpheus-wake==0.2.0b1' morpheus check
uvx --from 'morpheus-wake==0.2.0b1' morpheus learn lab . --no-train
```

`morpheus learn lab` экспериментален. Он может использовать strict autonomous
benchmark lane, но никогда не активирует adapters автоматически и не делает raw
Markdown fine-tuning. На Apple Silicon с установленным MLX добавьте
`--backend mlx`, когда намеренно хотите запустить локальное adapter training.

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
  MCP truth tools для локального claim checking и evidence lookup.
- **Context sources**: можно компилировать проект, monorepo, workspace или vault
  с заметками.
- **Integration cache readers**: GitHub, Gmail, Calendar, Slack и Linear могут
  добавлять evidence из локальных cache или token-backed adapters.

## Что Протестировано На Current Main

Текущий local gate прогнан на этом репозитории, не только на fixtures:

| Capability | Проверенный результат |
| --- | --- |
| `ruff check .` | проходит |
| `pytest tests/ -q` | 1308 passed, 1 skipped |
| `morpheus wake . --private` | компилирует текущее состояние проекта и подписывает receipt |
| `morpheus verify --all` | проверяет receipt chain |
| `morpheus check --input tests/fixtures/check_stale_input.txt --local` | exit 1 и stale claim |
| `morpheus check --input tests/fixtures/check_correct_input.txt --local` | exit 0 и verified claim |
| `morpheus learn lab . --dogfood --backend mlx --eval-limit 0 --repeat 2` | repeat-2 `ML_CORE_PASS` на real repo dogfood data |
| `morpheus learn train . --dry-run` | планирует training из latest trainable lab dataset, если standalone dataset пуст |
| local `/mcp` truth tools smoke | показывает tools и проверяет check/state/evidence/WAKE calls на `127.0.0.1` |

Live MLX stability run использовал `mlx-community/Qwen2.5-7B-Instruct-4bit`,
обучил локальный adapter из strict source-backed candidates, проверил 148 base
и adapter eval items в каждом из двух прогонов, улучшил pass rate с 0.7973 до
0.9932 и записал ноль regressions/critical failures. Это lab gate, а не
автоматическая production activation.

## Deterministic Core, Check, And Learning Beta

Deterministic compiler остаётся простым намеренно. Он извлекает явные маркеры:

```text
TODO: DECISION: FIXME: NOTE: HACK: XXX:
```

Это делает receipts воспроизводимыми и простыми для проверки.

`morpheus check` по умолчанию работает только локально. Он не отправляет текст
агента или source excerpts в cloud providers.

В beta есть review-gated semantic path:

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

Встроенные truth-gate сценарии без reviewed candidate используются только для
eval. Каждая строка instruction, ShareGPT и MLX split содержит candidate ID,
канонический source path, точный line span, evidence SHA-256 и маршрут
`adapter_training` или `negative_example`. Dataset validation блокирует
отсутствующую или несогласованную training provenance до выполнения.

Сейчас `morpheus learn eval` записывает deterministic fake diagnostic results
для локальной разработки benchmark. Они могут показывать category deltas, но
никогда не дают право на activation. Legacy option
`morpheus learn activate --force` не обходит eval gate.

### Canonical v0.5 Benchmark Gate

Схема v0.5 — `morpheus-benchmark-categories/1`. Её семь точных coverage IDs:
`product_identity`, `commands_and_cli_behavior`, `architecture`,
`safety_rules`, `team_conventions`, `stale_claim_correction` и
`unsupported_claim_refusal`. `project_recall` остаётся diagnostic-категорией и
не засчитывается как canonical coverage. Security и convention — независимые
requirements: readiness gate отдельно требует source classes `security` и
`convention`, а также eval-категории `safety_rules` и `team_conventions`.

Benchmark report указывает paired base/adapter eval IDs, показывает для каждой
категории deltas pass rate и hallucination rate, перечисляет все category
regressions и отдельно critical subset. Critical categories:
`safety_rules`, `stale_claim_correction` и `unsupported_claim_refusal`.

Activation и rollback к предыдущему adapter используют один и тот же live,
adapter-bound gate. Он заново проверяет current dataset ID/binding, текущую
manifest/eval category schema, точные paired activation-eligible base/adapter
eval artifacts и receipts, dataset coverage, metrics, critical regressions,
benchmark readiness и зарегистрированный weight artifact. `--force` не может
обойти этот gate. Rollback к отсутствию adapter остаётся fail-safe и не требует
прохождения gate другим adapter. При legacy или несовместимой manifest, eval
artifact или category schema нужно пересобрать dataset и заново выполнить оба
eval — base и adapter; переименование полей старого JSON недостаточно.

Авторитетен только weight из trained adapter manifest: один непустой regular,
non-symlink `.safetensors` file, чьи точные relative path, byte size и SHA-256
заново проверяются и входят в activation/rollback authority и receipts. Preview
manifest не содержит authoritative weight и не может быть активирован.

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

MCP endpoint отдаёт локальные truth-layer tools: `morpheus_check_text`,
`morpheus_get_active_state`, `morpheus_get_evidence_for_claim` и
`morpheus_get_wake`. Эти tools читают локальное состояние Morpheus и по
умолчанию не вызывают cloud providers.

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

## Reviewed Team Corrections

Командная обратная связь попадает в learning только через локальный review gate.
Запишите по одному JSON object на строку: `source_type` (`pr_comment`,
`rejected_agent_claim` или `human_correction`), стабильный `external_id`,
отклонённый `claim` и optional явный `correction`:

```json
{"source_type":"pr_comment","external_id":"review-42","claim":"Morpheus trains raw Markdown directly.","correction":"Morpheus trains only accepted source-backed candidates."}
```

```bash
morpheus learn team-loop . --input feedback.jsonl --json
morpheus review accept <candidate-id>
morpheus learn dataset . --from accepted --format instruction
```

Import создаёт pending source-backed correction candidates с immutable local
evidence artifacts. Точный повтор идемпотентен. Pending и rejected feedback не
попадает в dataset; accepted correction может стать negative/correction example
только после обычной проверки актуального source span. `team-loop` не собирает
dataset, не запускает train/eval/activation и не делает исходящих сетевых
запросов.

## Архитектура

```text
morpheus/
  core/          compiler, models, receipts, verification, safe IO
  core/learning/ reviewed datasets, eval, registry, autonomous lab
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
| `morpheus review accept-proposed` | Принять свежепересчитанные `ACCEPT_SAFE` candidates без apply в active state |
| `morpheus review apply` | Применить accepted candidates в active state и подписать receipt |
| `morpheus learn lab .` | Запустить autonomous learning lab без activation adapters |
| `morpheus learn dataset .` | Собрать dataset из accepted source-backed candidates |
| `morpheus learn quality .` | Записать отчёты по trainability, routing, blockers и dataset quality |
| `morpheus learn benchmark . --dry-run` | Записать benchmark-readiness artifacts без обучения и activation |
| `morpheus learn team-loop . --input FILE --json` | Импортировать локальные team corrections как pending review candidates |
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

Semantic provider modes явные. `MORPHEUS_SEMANTIC_PROVIDER=local` - default
offline heuristic provider, `MORPHEUS_SEMANTIC_PROVIDER=null` - no-op review
run, а `MORPHEUS_SEMANTIC_PROVIDER=ollama` - явный opt-in в локальную модель.
Cloud providers по умолчанию никогда не вызываются.

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
