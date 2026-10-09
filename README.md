# Claim Check

Курсовой проект по дисциплине «Многоагентные ИИ-системы» (AITU, магистратура).
Два агента проверяют политическое утверждение: один через LLM выделяет проверяемый тезис, второй прогоняет его через модель из моей дипломной работы ([nlp-disinformation-detection-aitu](https://github.com/nihsioK/nlp-disinformation-detection-aitu), RoBERTa + метаданные LIAR).

Это первый этап (Ассайнмент 2).
Вердикт по фактам система пока не выносит: модель даёт сигнал по формулировке и репутации спикера, а проверять факты по источникам будут следующие агенты.
Для утверждений вне американской политики аналитик помечает, что оценка модели не показательна.
Как всё устроено, простыми словами: [docs/how-it-works.md](docs/how-it-works.md).
Полная схема из семи агентов и таблица ролей: [docs/ARCHITECTURE.md](docs/ARCHITECTURE.md).

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

Код возврата `0`, если проверка завершилась, иначе `1`.

В Docker:

```bash
make up                     # docker compose up --build, порт 8000
make down
```

Файл модели `models/hybrid_liar/` и папка `data/` монтируются с хоста, поэтому образ лёгкий, а база SQLite переживает перезапуск.

## Модель и данные

| Путь | Что это | В git |
|---|---|---|
| `models/hybrid_liar/best_model.pt` | обученная модель из дипломного репозитория (476 МБ) | нет, скопировать из `nlp-disinformation-detection-aitu/models/hybrid_liar/` |
| `models/roberta-base/` | конфиг и токенизатор `roberta-base`, чтобы работать без сети | да |
| `data/liar/train.tsv` | LIAR train: история спикеров для ветки метаданных | да |

Загрузчик определяет архитектуру по размерам тензоров, так что подойдёт и новая версия модели.
Пути меняются переменными `HYBRID_CHECKPOINT`, `BASE_MODEL_DIR`, `LIAR_TRAIN_TSV`.

## LLM

Единственный провайдер - OpenRouter, через него доступны модели всех вендоров по одному ключу.
Сменить провайдера значит сменить строку модели: `anthropic/claude-sonnet-5.5`, `openai/gpt-5.4-mini`, `google/gemini-3.5-flash` и так далее.
Модель задаётся переменной `LLM_MODEL`, полем «Модель» на странице или флагом `--model`.
Ключ читается из окружения или `.env` и в журнал не пишется.

## Langfuse

Каждая проверка уходит в [Langfuse](https://langfuse.com) одним трейсом `fact_check`.
Вызов LLM записывает нативная интеграция `langfuse.openai`: модель, токены, стоимость от OpenRouter, сообщения и ошибки.
Агенты это наблюдения типа `agent` (`analyst`, `classifier`), поиск спикера - `retriever`, инференс модели - `tool`.
Вход трейса - утверждение, спикер и контекст, выход - статус, метка и уверенность.

Чтобы включить, нужны `LANGFUSE_PUBLIC_KEY`, `LANGFUSE_SECRET_KEY` и `LANGFUSE_BASE_URL` в `.env`, желательно ещё `LANGFUSE_TRACING_ENVIRONMENT` и `LANGFUSE_RELEASE`.
Без ключей трейсинг выключен, код не меняется.
Если ключи заданы, у запуска на странице появляется ссылка на трейс.

## Переменные окружения

| Переменная | Назначение | По умолчанию |
|---|---|---|
| `OPENROUTER_API_KEY` | ключ OpenRouter, обязателен | - |
| `LLM_MODEL` | модель OpenRouter | `anthropic/claude-sonnet-5.5` |
| `LANGFUSE_PUBLIC_KEY`, `LANGFUSE_SECRET_KEY`, `LANGFUSE_BASE_URL` | трейсинг, без ключей выключен | не заданы |
| `HYBRID_CHECKPOINT`, `BASE_MODEL_DIR`, `LIAR_TRAIN_TSV` | модель и данные | пути внутри репозитория |
| `DB_PATH` | файл SQLite | `data/app.db` |
| `RUN_TIMEOUT_S`, `LLM_TIMEOUT_S`, `LLM_MAX_RETRIES`, `MAX_STEPS` | лимиты | `120`, `45`, `2`, `8` |

## HTTP API

| Метод и путь | Назначение |
|---|---|
| `GET /api/config` | модель по умолчанию, подсказки, задан ли ключ, включён ли трейсинг |
| `POST /api/runs` | начать проверку: `{statement, speaker?, context?, model?}` -> `202 {run_id}` |
| `GET /api/runs` | последние запуски |
| `GET /api/runs/{run_id}` | состояние запуска, все события и доля вызовов по агентам |
| `GET /api/runs/{run_id}/events.jsonl` | журнал запуска в JSONL |

Проверка идёт в фоне, страница опрашивает `GET /api/runs/{run_id}` раз в 0.7 с.

## Лимиты и ошибки

- Пустое утверждение или длиннее 2000 символов отклоняется с `422` до запуска агентов.
- Каждый вызов LLM или инструмента идёт через одну функцию (`Run.call`): проверка времени, счётчик шагов, замер длительности, запись в журнал и в трейс.
- Повторы только на временные ошибки сети, `429` и `5xx`.
- Невалидный JSON от LLM: одна попытка исправить, потом статус `failed`.
- Конечные статусы: `completed`, `failed`, `timed_out`.
  Нет файла модели или провайдер недоступен - запуск заканчивается с понятной причиной, а не висит.

## Разработка

```bash
make test                   # pytest, без сети: заглушки LLM и модели
make lint                   # pre-commit: ruff check + format, whitespace, yaml, размер файлов
```

Хуки ставятся командой `make install` и срабатывают на каждый коммит.

## Структура

```
app/
  config.py          переменные окружения и лимиты
  messages.py        сообщения: Claim, Analysis, Prediction
  llm.py             клиент OpenRouter через langfuse.openai
  tools.py           дипломная модель и справочник спикеров LIAR
  agents.py          Analyst и Classifier
  runner.py          раннер: очередь агентов, лимиты, статусы, трейсинг
  store.py           SQLite: запуски и события, доля вызовов по агентам
  server.py          FastAPI и статика
  cli.py             запуск из командной строки
  static/index.html  страница
models/              roberta-base (в git) и hybrid_liar (локально)
data/liar/           LIAR train.tsv
tests/test_runner.py
docs/                how-it-works.md, ARCHITECTURE.md
Dockerfile, docker-compose.yml, Makefile, .pre-commit-config.yaml
```
