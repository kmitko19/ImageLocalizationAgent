# Image Localization Agent

Локальное приложение для анализа текста на изображениях, перевода и подготовки структурированного JSON для последующей локализации макета.

## 1. Overview

**Image Localization Agent** принимает изображение с текстом, распознаёт блоки, анализирует визуальный контекст, переводит текст и формирует `result.json` — контракт данных для будущего этапа отрисовки.

**Реализовано сейчас (MVP):**

- загрузка изображения через Gradio GUI;
- OCR (EasyOCR);
- Vision-анализ (OpenAI Vision) с использованием OCR blocks;
- OCR/Vision fusion с флагами review;
- visual analysis (OpenCV/Pillow);
- translation (OpenAI);
- Manual Review в GUI (Approve / Edit + Approve / Reject);
- сохранение `source.jpg` и `result.json` в `data/output/{source_image_id}/`;
- фиксированный набор test cases с offline evaluator и runner.

**Текущий MVP заканчивается на проверенном структурированном `result.json`.**

**Renderer пока НЕ реализован.** MVP не создаёт финальное локализованное изображение.

---

## 2. Requirements & Installation

### Требования

- Windows 10
- Python 3.12 (разработка и полный test suite проверены на Python 3.12.10; другие версии Python проектом не верифицированы)
- Интернет для OpenAI API (GUI / live pipeline) и первичной загрузки моделей EasyOCR
- Ключ `OPENAI_API_KEY` в `.env`

### Установка

```powershell
cd ImageLocalizationAgent
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
copy .env.example .env
```

Откройте `.env` и укажите свой ключ:

```
OPENAI_API_KEY=sk-...
```

Ключ можно получить на [platform.openai.com/api-keys](https://platform.openai.com/api-keys).

> При первом запуске EasyOCR скачает модели распознавания (~100–200 МБ). Это требует интернета и может занять несколько минут.

### `config/settings.yaml`

Файл `config/settings.yaml` задаёт несекретные параметры приложения:

- модели OpenAI (`vision_model`, `translation_model`);
- языки OCR и списки языков для GUI;
- пороги fusion (`iou_threshold`, `text_mismatch_threshold`, `low_ocr_confidence_threshold` и др.);
- пути `data/input/test_cases` и `data/output`;
- порт GUI (`7860`).

Секреты (API key) загружаются только из `.env` через `config.py`.

---

## 3. Running the GUI

Из корня проекта в PowerShell:

```powershell
$env:PYTHONPATH="src"
.\.venv\Scripts\python.exe -m imageloc.app
```

Приложение откроется в браузере на **http://127.0.0.1:7860**.

> Не используйте устаревшую команду `python -m src.imageloc.app` — пакет импортируется как `imageloc` при `PYTHONPATH=src`.

---

## 4. GUI Workflow

Фактический порядок работы в интерфейсе:

```
Upload image
→ Source language
→ Target language
→ Analyze
→ Preview
→ Result JSON
→ Manual Review (при необходимости)
```

### Analyze

1. Загрузите изображение (PNG/JPG/WebP) через **Upload image**.
2. Выберите **Source language** (`auto`, `ru`, `en`) и **Target language**.
3. Нажмите **Analyze**. Обработка может занять 30–90 секунд (OpenAI Vision + Translation + локальный EasyOCR).
4. Проверьте **Preview (bbox + block IDs)** — превью с рамками и номерами блоков.
5. Изучите **Result JSON** — полный результат анализа.

### Manual Review

Если fusion пометил блоки как `requires_review`, используйте секцию **Manual Review**:

- выберите блок в **Blocks requiring review**;
- просмотрите OCR raw text, Vision corrected text, review reasons, confidence, bbox и crop;
- **Approve** — подтвердить блок без правок;
- **Edit + Approve** — изменить original/translated text и подтвердить;
- **Reject** — отклонить блок (кнопка присутствует в GUI).

После каждого действия интерфейс переходит к следующему pending-блоку. В **Review progress** отображается:

- `Reviewed N / N`
- `Remaining M`
- при завершении очереди: `Review complete`, `Reviewed N / N`, `Remaining 0`

### Result JSON

- JSON отображается в отдельном scrollable окне (**Result JSON**).
- Длинный JSON не растягивает интерфейс (фиксированная высота области прокрутки).
- Отдельной кнопки **Download** в GUI **нет** — JSON доступен на экране и сохраняется на диск автоматически.

> В GUI нет test-case picker и нет таблицы результатов — только upload, preview, JSON и Manual Review.

---

## 5. Actual Pipeline

Фактическая последовательность обработки (не parallel):

```
image
→ OCR (EasyOCR)
→ Vision (OpenAI), с использованием OCR blocks
→ OCR/Vision fusion
→ visual analysis (OpenCV/Pillow)
→ translation (OpenAI)
→ review workflow (fusion flags + GUI Manual Review)
→ result.json
```

**Renderer в реализованный pipeline не входит.**

---

## 6. Manual Review & JSON

Каждый текстовый блок в `result.json` содержит объект `review` со следующими полями:

| Поле | Описание |
|------|----------|
| `requires_review` | `true`, если блок требует ручной проверки |
| `reasons` | список причин review (см. ниже) |
| `ocr_vision_similarity` | сходство OCR/Vision (если вычислено) |
| `user_confirmed` | `true` / `false` / `null` — решение пользователя |
| `user_action` | `"approved"`, `"edited"`, `"rejected"` или `null` |

Блоки без `requires_review` автоматически получают `user_confirmed=true` и `user_action="approved"` при формировании JSON.

### Review reasons (фактические значения в `review.reasons`)

- `low_ocr_confidence` — низкая уверенность EasyOCR;
- `text_mismatch` — расхождение текста OCR и Vision;
- `probable_non_text` — вероятный non-text элемент (графика, а не текст);
- `vision_scope_fallback` — Vision вернул fallback в рамках scope guard.

---

## 7. Output

После **Analyze** для загруженного изображения создаётся каталог:

```
data/output/{source_image_id}/
├── source.jpg      — копия исходного изображения
└── result.json     — структурированный результат pipeline
```

- **Preview** генерируется только для GUI (bbox + block IDs) и **не сохраняется** на диск как отдельный файл.
- После каждого действия **Manual Review** (Approve / Edit + Approve / Reject) `result.json` **перезаписывается** на диск с обновлёнными полями `review` (и при Edit + Approve — с обновлёнными текстами блока).

Каталог `data/output/` исключён из Git (`.gitignore`).

---

## 8. Test Cases

Фиксированный набор из 5 test cases в `data/input/test_cases/`:

| Case ID | Image | Статус |
|---------|-------|--------|
| `01_complex_brochure_ru` | `01_complex_brochure_ru.jpg` | **available** |
| `02_text_and_nontext_ru` | `02_text_and_nontext_ru.jpg` | **available** |
| `03_text_on_photo_ru` | `03_text_on_photo_ru.jpg` | **MISSING** |
| `04_decorative_headline_ru` | `04_decorative_headline_ru.jpg` | **MISSING** |
| `05_catalog_mixed_layout_ru` | `05_catalog_mixed_layout_ru.jpg` | **MISSING** |

Структура:

- `baselines/` — агрегированные метрики baseline (отдельно от acceptance);
- `acceptance/` — критерии приёмки (`*.acceptance.json`);
- `*.jpg.MISSING` — текстовые маркеры для отсутствующих изображений 03–05.

Offline evaluator (`src/imageloc/utils/test_cases.py`) сравнивает сохранённые E2E reference results с baseline/acceptance **без вызовов OpenAI**.

### Offline runner

```powershell
$env:PYTHONPATH="src"
.\.venv\Scripts\python.exe scripts/run_test_cases.py
```

Offline-режим (по умолчанию):

- **не требует** `OPENAI_API_KEY`;
- **не вызывает** OpenAI или другие внешние API;
- использует сохранённые E2E/baseline данные из `data/output/.../result.json`.

Контрольная точка на момент завершения To-do №12 (snapshot, не гарантия на будущее): **2 PASS / 3 SKIP / 0 FAIL**.

### `--run-api` (explicit opt-in)

```powershell
$env:PYTHONPATH="src"
.\.venv\Scripts\python.exe scripts/run_test_cases.py --run-api
```

Запускает live pipeline (EasyOCR + OpenAI). Требует `OPENAI_API_KEY`. **Не является** обычным или обязательным режимом — только явный opt-in.

Подробности по каждому кейсу — в `data/input/test_cases/README.md`.

---

## 9. Development & Tests

Все команды — из корня проекта с `PYTHONPATH=src`:

```powershell
$env:PYTHONPATH="src"
.\.venv\Scripts\python.exe -m pytest
```

Отдельно test-case integration tests:

```powershell
$env:PYTHONPATH="src"
.\.venv\Scripts\python.exe -m pytest tests\test_test_cases_integration.py -v
```

Контрольная точка на момент завершения To-do №12: **103 passed, 0 failed** (полный pytest). Это snapshot текущего состояния, а не постоянное свойство проекта.

### Структура проекта

```
src/imageloc/
├── app.py                 — Gradio GUI
├── config.py              — .env + settings.yaml
├── models/                — Pydantic schema (JSON v1.1)
├── providers/             — OCR / Vision / Translation
├── fusion/                — OCR/Vision fusion
├── pipeline/              — orchestrator
└── utils/                 — image loader, visual analyzer, JSON export,
                           preview, review workflow, test cases

config/settings.yaml
data/input/test_cases/     — test images, baselines, acceptance
data/output/               — результаты анализа (не в Git)
scripts/run_test_cases.py  — offline/API test-case runner
tests/                     — unit и integration tests
```

---

## 10. Security

- `OPENAI_API_KEY` хранится только в `.env`.
- `.env` исключён из Git (`.gitignore`).
- Секреты не должны попадать в код, JSON-результаты, логи или GUI.
- Offline test-case runner (`scripts/run_test_cases.py` без `--run-api`) **не требует** API key.

---

## 11. MVP Limitations

### Текущий MVP умеет

- принимать изображение;
- выполнять OCR;
- выполнять Vision analysis;
- сопоставлять OCR/Vision (fusion);
- выполнять visual analysis;
- переводить текст;
- проводить Manual Review проблемных блоков;
- формировать и сохранять структурированный `result.json`.

### Текущий MVP пока НЕ умеет

- удалять исходный текст с изображения;
- выполнять inpainting / восстановление фона;
- отрисовывать перевод на изображении;
- подбирать финальную типографику для рендера;
- создавать финальное локализованное изображение.

---

## 12. Roadmap

**FUTURE / NOT IMPLEMENTED**

Следующий крупный этап — **Renderer**.

Планируемый pipeline после реализации Renderer:

```
reviewed result.json
→ удаление исходного текста
→ восстановление фона
→ размещение перевода
→ typography/layout fitting
→ финальное локализованное изображение
```

Renderer, inpainting, batch-обработка и расширенный QA — отдельные будущие этапы после текущего MVP.
