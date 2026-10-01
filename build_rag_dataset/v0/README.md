# Автономная сборка датасета v0

Эта директория содержит воспроизводимый pipeline для сборки датасета `hf/v0`
без зависимости от исходной папки `rag/`. Pipeline поддерживает два режима:

- полная генерация из переданного каталога изображений через Cursor SDK;
- дешёвая offline-пересборка из зафиксированных producer/verifier-ответов.

## Исходная точка

В полном режиме исходной точкой являются изображения, image-label contracts,
ontology и prompts. Каталог изображений не хранится здесь и передаётся через
placeholder `--images-root /path/to/pilot-500-v1`. Pipeline выполняет:

1. генерацию структурированных описаний моделью producer;
2. независимую проверку каждого image-label fact моделью verifier;
3. producer/verifier agreement;
4. создание normalized facts;
5. trusted/untrusted partition;
6. ontology catalog;
7. one-fact-per-chunk представление;
8. manifest и итоговую валидацию.

Для воспроизводимой проверки без повторной оплаты VLM-вызовов исходные ответы
producer/verifier также сохранены в `input/`. Готовые `facts_*.jsonl`, chunks и
agreement summary из `hf/v0` в качестве входов не используются. Финальные
producer/verifier records входят в release как annotations; остальные
представления создаются в `output/`.

## Содержимое

### `input/`

Неизменяемые исходные данные сборки:

- `annotations/descriptions.jsonl` — зафиксированные producer-ответы для
  offline-режима;
- `annotations/verifications.jsonl` — зафиксированные verifier-ответы для
  offline-режима;
- `annotations/image_label_contracts.jsonl` — исходные и effective labels,
  policy и provenance для этих изображений;
- `ontology/labels.yaml` — полный ontology из 136 категорий;
- `prompts/producer.md`, `prompts/verifier.md` — использованные prompts;
- `config/models.yaml` — конфигурация моделей;
- `config/build.yaml` — идентификаторы запусков, параметры release и
  документированный placeholder `images_root`;
- `provenance/images.jsonl` — только пути, размеры и SHA-256 изображений; сами
  изображения сюда не входят;
- `dataset_README.md` — шаблон dataset card итогового release.

Файлы отфильтрованы до producer/verifier-запусков pilot-500-v1. Другие
экспериментальные запуски из `rag/records` не копировались.

### Скрипты

- `generate_descriptions.py` передаёт каждое изображение, ontology и
  разрешённые labels producer-модели через Cursor SDK и валидирует
  структурированный JSON;
- `verify_descriptions.py` передаёт изображение и producer-output независимой
  verifier-модели через Cursor SDK и валидирует verdicts;
- `cursor_vlm.py` реализует image message, выбор модели, JSON parsing, retries,
  cache и SDK provenance для двух VLM-этапов;
- `build_dataset.py` вычисляет agreement из выбранных producer/verifier
  records, нормализует image-label facts, создаёт
  trusted/untrusted partition, ontology catalog, provenance и manifest;
- `build_fact_chunks.py` преобразует каждый normalized fact в один retrieval
  chunk и создаёт варианты `all` и `trusted`;
- `validate_dataset.py` проверяет manifest counts, уникальность facts,
  trusted/untrusted partition, chunk-to-fact correspondence и синтаксис ссылок
  на изображения. Наличие самих изображений не требуется;
- `build_all.py` запускает три предыдущих этапа в правильной
  последовательности и останавливается при первой ошибке.

## Установка

Из корня `e2e_rag_gigaschool_llm_course`:

```bash
python -m pip install -e .
```

## Offline-сборка без вызова моделей

```bash
python build_rag_dataset/v0/build_all.py
```

Результат появится в `build_rag_dataset/v0/output/` и будет иметь ту же
структуру, что и `hf/v0`:

```text
output/
├── annotations/
│   ├── gt/
│   └── vlm_inferred/
├── offline_chunking/
├── prompts/
├── provenance/
├── manifest.json
└── README.md
```

Для публикации результат можно скопировать в `hf/v0` только после проверки
изменений. Builder никогда не изменяет `hf/v0` самостоятельно.

## Полная сборка из изображений

Нужен `CURSOR_API_KEY`, а модели из `input/config/models.yaml` должны быть
доступны аккаунту:

```bash
export CURSOR_API_KEY="cursor_..."

python build_rag_dataset/v0/build_all.py \
  --generate-vlm \
  --images-root /path/to/pilot-500-v1
```

Placeholder `/path/to/pilot-500-v1` должен указывать на каталог, внутри
которого находятся пути из `input/provenance/images.jsonl`, например
`100/27-5Donskova3.png`.

Producer и verifier используют разные модели. Успешные ответы записываются
сразу в финальные
`output/annotations/vlm_inferred/{descriptions,verifications}.jsonl`.
SDK cache хранится в `.cache/` и не входит в release или Git.

Прерванный дорогой запуск можно продолжить с теми же run IDs:

```bash
python build_rag_dataset/v0/build_all.py \
  --generate-vlm \
  --resume \
  --images-root /path/to/pilot-500-v1
```

## Минимальный smoke test

Для проверки кода на двух изображениях и трёх labels каждого:

```bash
python build_rag_dataset/v0/build_all.py \
  --max-images 2 \
  --max-labels-per-image 3
```

Эта команда использует зафиксированные VLM-ответы и не вызывает Cursor SDK.
Для платного end-to-end smoke test добавьте `--generate-vlm`, `--images-root`
и при необходимости собственные run IDs. Smoke output не является публикуемой
версией датасета.

## Пользовательские пути

Все этапы поддерживают отдельный output:

```bash
python build_rag_dataset/v0/build_all.py \
  --input-root build_rag_dataset/v0/input \
  --output /tmp/video-retrieval-v0
```

`build_all.py` очищает выбранный output перед новым запуском. Исключение —
полный режим с `--resume`, который сохраняет уже полученные VLM records.

