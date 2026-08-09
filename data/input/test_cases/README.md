# Test Cases

Фиксированный набор из **5** test cases для регрессии и offline-проверки MVP **до** этапа Renderer.

Baseline и acceptance — **раздельные** сущности. Mock-изображения для отсутствующих кейсов **не создаются**.

---

## Каталог кейсов

| Case ID | Image file | Статус | Назначение |
|---------|------------|--------|------------|
| `01_complex_brochure_ru` | `01_complex_brochure_ru.jpg` | **available** | Сложная брошюра: много мелкого русского текста, сложный фон; проверка OCR/Vision matching, fusion, translation, review |
| `02_text_and_nontext_ru` | `02_text_and_nontext_ru.jpg` | **available** | Текст + отдельный графический non-text элемент (heart-image); проверка `probable_non_text`, `low_ocr_confidence`, Manual Review |
| `03_text_on_photo_ru` | `03_text_on_photo_ru.jpg` | **MISSING** | Текст поверх неоднородной фотографии; OCR/Vision на сложном фоне |
| `04_decorative_headline_ru` | `04_decorative_headline_ru.jpg` | **MISSING** | Крупный декоративный / стилизованный заголовок |
| `05_catalog_mixed_layout_ru` | `05_catalog_mixed_layout_ru.jpg` | **MISSING** | Смешанный каталожный layout |

### Текущее состояние изображений

- **01 / 02** — реальные JPG присутствуют в этом каталоге (скопированы из контрольных E2E прогонов).
- **03–05** — изображений **нет**. Вместо mock-файлов используются маркеры `03_text_on_photo_ru.jpg.MISSING` и аналогичные для 04/05.

---

## Структура каталога

```
data/input/test_cases/
├── 01_complex_brochure_ru.jpg
├── 02_text_and_nontext_ru.jpg
├── 03_text_on_photo_ru.jpg.MISSING
├── 04_decorative_headline_ru.jpg.MISSING
├── 05_catalog_mixed_layout_ru.jpg.MISSING
├── baselines/
│   ├── 01_complex_brochure_ru.baseline.json
│   └── 02_text_and_nontext_ru.baseline.json
├── acceptance/
│   ├── 01_complex_brochure_ru.acceptance.json
│   ├── 02_text_and_nontext_ru.acceptance.json
│   ├── 03_text_on_photo_ru.acceptance.json
│   ├── 04_decorative_headline_ru.acceptance.json
│   └── 05_catalog_mixed_layout_ru.acceptance.json
└── README.md
```

### `baselines/`

Агрегированные метрики, извлечённые из контрольных E2E `result.json` (без повторного вызова API):

- `stats` (total_blocks, ocr_blocks, vision_blocks, blocks_requiring_review, …);
- `review_reason_counts`;
- `source_e2e_result` — путь к reference E2E в `data/output/.../result.json`;
- контрольные `required_texts` и `required_blocks`.

Baseline **не** дублирует полный `result.json`.

### `acceptance/`

Критерии offline-проверки для каждого case ID:

- `image_status`: `"available"` или `"missing"`;
- `baseline_file` — ссылка на baseline (для 01/02) или `null` (для 03–05);
- `automated_run`: `true` для available, `false` для missing;
- `expectations` — пороги stats, review reason counts, required texts/blocks.

### `*.jpg.MISSING`

Текстовые placeholder-файлы для кейсов без реального изображения. Runner и evaluator трактуют такие кейсы как **SKIP**, а не **FAIL**.

---

## Offline evaluator

Модуль `src/imageloc/utils/test_cases.py`:

- загружает acceptance/baseline manifests;
- для available-кейсов читает reference E2E `result.json` с диска;
- проверяет метрики и acceptance expectations;
- **не вызывает** OpenAI или другие внешние API.

---

## Runner

```powershell
$env:PYTHONPATH="src"
.\.venv\Scripts\python.exe scripts/run_test_cases.py
```

По умолчанию — **полностью offline**:

- обнаруживает все 5 test cases;
- для 01/02 запускает offline evaluator → **PASS** / **FAIL**;
- для 03–05 → **SKIP** (missing image);
- **не требует** `OPENAI_API_KEY`.

Контрольная точка (snapshot): **2 PASS / 3 SKIP / 0 FAIL**.

### `--run-api` (explicit opt-in)

```powershell
$env:PYTHONPATH="src"
.\.venv\Scripts\python.exe scripts/run_test_cases.py --run-api
```

Live pipeline через orchestrator (`is_test_case=True`). Требует `OPENAI_API_KEY`. Не является обычным или обязательным режимом.

---

## Как добавить отсутствующий test image (03–05)

Когда появится реальный эталонный снимок:

1. Поместите файл с **точным** именем из таблицы (например `03_text_on_photo_ru.jpg`) в `data/input/test_cases/`.
2. Удалите соответствующий `*.jpg.MISSING` marker.
3. Выполните live E2E прогон (GUI или `--run-api`) и сохраните контрольный `result.json`.
4. Сгенерируйте baseline и acceptance через `scripts/_generate_test_case_assets.py` (или обновите metadata вручную по принятому формату).
5. Запустите offline runner для проверки.

**Не используйте mock-изображения** — только реальные эталонные снимки для целевого сценария.

---

## Ожидания для MVP (available cases)

- **01** — сложная брошюра: Vision исправляет OCR, fusion выставляет review flags, все блоки переведены.
- **02** — текст + non-text: наличие `probable_non_text`, `low_ocr_confidence`, корректная Manual Review очередь.
- Блоки с `requires_review=true` должны быть доступны в GUI для Approve / Edit + Approve / Reject.
