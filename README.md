# Claim Check

Курсовой проект по дисциплине «Многоагентные ИИ-системы» (AITU, магистратура).
Шесть агентов проверяют политическое утверждение: аналитик выделяет тезис, поиск собирает источники, классификатор из моей дипломной работы ([nlp-disinformation-detection-aitu](https://github.com/nihsioK/nlp-disinformation-detection-aitu), RoBERTa + метаданные LIAR) даёт свой сигнал, оценщик взвешивает источники, синтезатор пишет вердикт со ссылками, а проверяющий без LLM принимает его или один раз возвращает на доработку.

Как всё устроено, простыми словами: [docs/how-it-works.md](docs/how-it-works.md).
Схема, таблица ролей, форматы сообщений и лимиты: [docs/ARCHITECTURE.md](docs/ARCHITECTURE.md).
Примеры для проверки: [docs/examples.xlsx](docs/examples.xlsx).

## Запуск

Нужны Python 3.12 и [uv](https://docs.astral.sh/uv/).

```bash
make install                # uv sync + pre-commit hooks
cp .env.example .env        # вписать OPENROUTER_API_KEY
make dev                    # http://127.0.0.1:8000
```

Из командной строки:

```bash
uv run python -m app.cli "Each year, 18,000 people die in America because they don't have health care." --speaker "Hillary Clinton"
```

Код возврата `0`, если проверка завершилась вердиктом, иначе `1`.

В Docker:

```bash
make up                     # docker compose up --build, порт 8000
make down
```

Файл модели `models/hybrid_liar/` и папка `data/` монтируются с хоста, поэтому образ лёгкий, а база SQLite переживает перезапуск.

## Как проходит проверка

1. **Analyst** (LLM) превращает вход в один проверяемый тезис, отмечает область (американская политика или нет), язык и поисковые запросы.
2. Параллельно: **Retriever** ищет источники в вебе и сохраняет фрагменты как есть, **Classifier** ищет спикера в LIAR и прогоняет тезис через модель диплома.
3. **Assessor** (LLM + правила по доменам) ставит каждому источнику релевантность, надёжность и позицию: поддерживает, опровергает, смешанно, не по теме.
4. **Synthesizer** (LLM) пишет вердикт (`true`, `mostly-true`, `half-true`, `mostly-false`, `false`, `unverifiable`) с уверенностью и ссылками только на существующие источники.
5. **Verifier** (правила, без LLM) проверяет цитаты, согласованность вердикта с позициями источников, границы уверенности и доступность ссылок. Один круг исправлений, второй отказ завершает запуск ошибкой.

Раннер связывает агентов, ведёт лимиты и журнал, сам ничего не анализирует.
Если источников не нашлось, запуск заканчивается статусом `insufficient_evidence` без выдуманных ссылок.

## Модель и данные

| Путь | Что это | В git |
|---|---|---|
| `models/hybrid_liar/best_model.pt` | обученная модель из дипломного репозитория (476 МБ) | нет, скопировать из `nlp-disinformation-detection-aitu/models/hybrid_liar/` |
| `models/roberta-base/` | конфиг и токенизатор `roberta-base`, чтобы работать без сети | да |
| `data/liar/train.tsv` | LIAR train: история спикеров для ветки метаданных | да |

Загрузчик определяет архитектуру по размерам тензоров, так что подойдёт и новая версия модели.
Модель знает формулировку и репутацию спикера, но не факты: её сигнал участвует в вердикте только как один из входов, а вне американской политики помечается как непоказательный.

## LLM и поиск

Единственный провайдер LLM - OpenRouter, через него доступны модели всех вендоров по одному ключу.
Сменить провайдера значит сменить строку модели: `anthropic/claude-sonnet-5.5`, `openai/gpt-5.4-mini`, `google/gemini-3.5-flash` и так далее.
Модель задаётся переменной `LLM_MODEL`, полем «Модель» на странице или флагом `--model`.

Поиск источников: по умолчанию веб-плагин OpenRouter (тот же ключ), при наличии `TAVILY_API_KEY` используется Tavily.
Результаты кэшируются в SQLite на сутки.
Режим `SEARCH_BACKEND=cache` отвечает только из кэша, это способ показать систему без сети: прогнать нужные утверждения заранее, а на защите повторить.
Ключи читаются из окружения или `.env` и в журнал не пишутся.

## Langfuse

Каждая проверка уходит в [Langfuse](https://langfuse.com) одним трейсом `fact_check`.
Вызовы LLM записывает нативная интеграция `langfuse.openai`: модель, токены, стоимость от OpenRouter, сообщения и ошибки.
Агенты это наблюдения типа `agent`, поиск спикера и веб-поиск - `retriever`, инференс модели, правила доменов и проверка ссылок - `tool`.
Вход трейса - утверждение, спикер и контекст, выход - статус, вердикт и уверенность.

Чтобы включить, нужны `LANGFUSE_PUBLIC_KEY`, `LANGFUSE_SECRET_KEY` и `LANGFUSE_BASE_URL` в `.env`, желательно ещё `LANGFUSE_TRACING_ENVIRONMENT` и `LANGFUSE_RELEASE`.
Без ключей трейсинг выключен, код не меняется.

## Переменные окружения

| Переменная | Назначение | По умолчанию |
|---|---|---|
| `OPENROUTER_API_KEY` | ключ OpenRouter, обязателен | - |
| `LLM_MODEL` | модель OpenRouter | `anthropic/claude-sonnet-5.5` |
| `SEARCH_BACKEND` | `openrouter`, `tavily` или `cache` | `tavily`, если задан ключ, иначе `openrouter` |
| `TAVILY_API_KEY`, `SEARCH_RESULTS`, `SEARCH_MODEL` | ключ Tavily, число результатов, модель для поискового вызова плагина | не задан, `5`, `openai/gpt-5.4-nano` |
| `LANGFUSE_PUBLIC_KEY`, `LANGFUSE_SECRET_KEY`, `LANGFUSE_BASE_URL` | трейсинг, без ключей выключен | не заданы |
| `HYBRID_CHECKPOINT`, `BASE_MODEL_DIR`, `LIAR_TRAIN_TSV` | модель и данные | пути внутри репозитория |
| `DB_PATH` | файл SQLite | `data/app.db` |
| `RUN_TIMEOUT_S`, `LLM_TIMEOUT_S`, `LLM_MAX_RETRIES`, `MAX_STEPS` | лимиты | `180`, `45`, `2`, `20` |

## HTTP API

| Метод и путь | Назначение |
|---|---|
| `GET /api/config` | модель по умолчанию, подсказки, задан ли ключ, поисковый бэкенд, включён ли трейсинг |
| `POST /api/runs` | начать проверку: `{statement, speaker?, context?, model?}` -> `202 {run_id}` |
| `GET /api/runs` | последние запуски |
| `GET /api/runs/{run_id}` | состояние запуска, все события и доля вызовов по агентам |
| `GET /api/runs/{run_id}/events.jsonl` | журнал запуска в JSONL |

Проверка идёт в фоне, страница опрашивает `GET /api/runs/{run_id}` раз в 0.7 с.

## Лимиты и ошибки

- Пустое утверждение или длиннее 2000 символов отклоняется с `422` до запуска агентов.
- Каждый вызов LLM или инструмента идёт через одну функцию (`Run.call`): проверка времени, счётчик шагов, замер длительности, запись в журнал и в трейс.
- Повторы только на временные ошибки сети, `429` и `5xx`.
- Невалидный JSON от LLM или ссылка на несуществующий источник: одна попытка исправить, потом статус `failed`.
- Фрагменты страниц считаются данными: обрезаются до 600 символов и передаются в промпты с пометкой «недоверенный текст».
- Конечные статусы: `completed`, `failed`, `timed_out`, `insufficient_evidence`.

## Разработка

```bash
make test                   # pytest, без сети: заглушки LLM, поиска и модели
make lint                   # pre-commit: ruff check + format, whitespace, yaml, размер файлов
```

Тесты покрывают сценарии: типовой прогон со всеми агентами и долей вызовов не выше 40 %, опровергающие источники, отсутствие источников, недоступный поиск, один круг revise, двойной отказ проверяющего, вердикт против позиции источников, инструкции внутри найденного текста, невалидный JSON, отсутствующий файл модели, истёкший дедлайн, утверждение вне домена, кэш поиска, API.

## Структура

```
app/
  config.py          переменные окружения и лимиты
  messages.py        сообщения: Claim, Analysis, Evidence, Prediction, Assessment, Draft, Verification
  llm.py             клиент OpenRouter через langfuse.openai, JSON с одной попыткой починки
  tools.py           модель диплома, справочник спикеров LIAR, веб-поиск с кэшем, правила доменов, проверка ссылок
  agents.py          Analyst, Retriever, Classifier, Assessor, Synthesizer, Verifier
  runner.py          раннер: очередь агентов, параллельный шаг, revise, лимиты, статусы, трейсинг
  store.py           SQLite: запуски, события, кэш поиска, доля вызовов по агентам
  server.py          FastAPI и статика
  cli.py             запуск из командной строки
  static/index.html  страница
models/              roberta-base (в git) и hybrid_liar (локально)
data/liar/           LIAR train.tsv
tests/test_runner.py
docs/                how-it-works.md, ARCHITECTURE.md, examples.xlsx, report_assignment_2.docx
Dockerfile, docker-compose.yml, Makefile, .pre-commit-config.yaml
```
