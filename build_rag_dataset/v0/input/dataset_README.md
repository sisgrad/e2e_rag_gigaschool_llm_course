# Camera Image RAG v0

Версия `v0` — воспроизводимый срез из 500 изображений городских камер для курса
по end-to-end RAG. Для каждого изображения сохранены исходный структурированный
ответ producer-модели, независимая проверка verifier-моделью и нормализованные
факты с признаком доверия.

## Как получена разметка

1. Исходные positive labels взяты из `base_static` и `detailed_static`.
2. Producer `gpt-5.6-sol` видел изображение и полный ontology, но мог описывать
   только labels из whitelist конкретного изображения.
3. Producer возвращал общий summary и массив `facts`, где каждый факт содержит
   `label_id`, `sentence`, `evidence` и `visibility`.
4. Независимый verifier `claude-opus-5-5` повторно видел изображение и для каждого
   факта определял `supported`, `uncertain` или `unsupported`, а также проверял
   faithful wording/evidence.
5. Факт получает `trusted_fact=true` только при состоянии `agreed_good`.

Разные семейства моделей уменьшают риск коррелированных ошибок. OCR, timestamp,
watermark и надписи не принимались как визуальное доказательство категории.
Отсутствующая категория означает `unjudged`, а не отрицательный пример.

## Уровни labels

- `base_static` (`label_level=base`) — 43 наблюдавшиеся базовые категории:
  положение камеры, тип сцены, время суток, погода и качество изображения.
- `detailed_static` (`label_level=detailed`) — 92 наблюдавшиеся детальные
  категории: дорожная ситуация, объекты инфраструктуры, здания и другие признаки.
- Одна ontology-категория не встретилась в pilot 500.

Полный список, определения, entities и частоты находятся в
`annotations/gt/ontology/label_catalog.json` и
`annotations/gt/ontology/labels.yaml`.

## Размер

- 500 изображений;
- 500 исходных producer-записей;
- 500 verifier-записей;
- 7 481 уникальный `(image, label)` факт;
- 5 264 доверенных факта;
- 2 217 недоверенных/неопределённых фактов;
- 136 ontology labels, из них 135 наблюдались.

## Структура

### `../images/pilot-500-v1/`

500 исходных изображений вынесены из версионной папки `v0` и разделяются между
версиями разметки. Локально файлы создаются hardlink-ами, поэтому повторно не
занимают место; при архивировании становятся обычными файлами.

### `annotations/gt/image_labels.jsonl`

Исходный whitelist human labels и effective policy для каждого изображения.
Сохраняет разделение на `base_static` и `detailed_static`.

### `annotations/gt/ontology/`

Полный ontology и компактный каталог labels с делением base/detailed.

### `annotations/vlm_inferred/descriptions.jsonl`

Одна строка — полный исходный VLM producer-ответ для одного изображения, без
чанкинга:

- `image_id`, `image_rel_path`, `allowed_labels`;
- `description.description` — общий image-level summary;
- `description.facts[]` — структурированные sentence/evidence;
- `validation` — schema/whitelist проверки;
- `producer` — model, SDK version, run IDs, usage;
- `provenance_hash`.

`allowed_labels` — не глобальный список ontology. Это whitelist положительных
labels из исходной разметки конкретного изображения, переданный producer. Он
различается между изображениями и запрещает producer добавлять отсутствующие в
исходной разметке категории. Поэтому pipeline проверяет описание известных
positives, но не ищет пропущенные в GT объекты: такие false negatives остаются
`unjudged`.

Это основной raw-корпус для EDA контрольной точки 1.

### `annotations/vlm_inferred/verifications.jsonl`

Полный исходный verifier-ответ на изображение:

- `verification.label_verdicts[]` — verdict и rationale по каждой категории;
- `all_labels_covered`, `invented_detail`, `out_of_whitelist_mention`;
- `overall`, `notes`;
- validation и SDK provenance.

### `annotations/vlm_inferred/facts_all.jsonl`

Нормализованная таблица уникальных фактов. Одна строка — один `v0-fact-1`:

- исходные `sentence`, `evidence`, `visibility`;
- label, layer и entity;
- verifier verdict/rationale;
- `agreement_state`, `trusted_fact`, `trust_reasons`;
- число исходных дубликатов и run provenance.

Это разметка, а не retrieval chunks.

### `annotations/vlm_inferred/facts_trusted.jsonl`

Подмножество `facts_all` с `trusted_fact=true`.

### `annotations/vlm_inferred/facts_untrusted.jsonl`

Факты со статусами `verifier_uncertain`, `agreed_bad`, `producer_only` или
`failed`. Они не удалены и доступны для аудита и альтернативных политик.

### `offline_chunking/`

Производное, полностью регенерируемое представление:

- `fact_chunks_all.jsonl`;
- `fact_chunks_trusted.jsonl`.

Граница chunk определяется не эвристикой по свободному тексту, а элементом
`facts[]` структурированного producer-ответа. Один факт становится одним chunk;
semantic splitting и overlap не применяются. Исходные тексты всегда остаются в
`annotations/vlm_inferred/`.

### `prompts/`

Точные producer/verifier prompts, использованные при генерации.

### `provenance/`

Model configuration и итоговая agreement statistics. Идентификаторы запусков
сохранены непосредственно в VLM-разметке и agreement summary.

### `manifest.json`

Краткое вручную поддерживаемое описание версии schema, counts и semantics.

EDA и evaluation намеренно не хранятся внутри dataset:

- `../../eda/v0/output/`;
- `../../evaluation/v0/`.

## Воспроизведение

Из корня `base_n_detailed_dataset_in_a_new_format`:

```bash
python -m pip install -e e2e_rag_gigaschool_llm_course
python e2e_rag_gigaschool_llm_course/build_rag_dataset/v0/build_all.py
```

Скрипты этой версии явно используют контракт
`gigaschool_rag.versions.v0`; символическое `vX` в общей документации означает
версию вида `v0`, `v1`, …, а не буквальное имя директории.
Автономная сборка записывает результат в
`build_rag_dataset/v0/output/` и не изменяет опубликованный `hf/v0`.

## Ограничение evidence

Высокие TF-IDF retrieval metrics на fact chunks показывают, что структурированный
формат хорошо индексируется и позволяет детерминированно восстановить границы
фактов. Они **не доказывают**, что такой подход лучше free-form descriptions:
canonical category может встречаться прямо в query/chunk. Для такого вывода нужен
отдельный A/B-эксперимент с одинаковыми вопросами, embedding model и retrieval
budget для structured и free-form вариантов.
