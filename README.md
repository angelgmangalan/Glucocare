# GlucoCare

**GlucoCare: A Personalized Multimodal Generative AI Assistant for Diabetes Care and Regional Diet Management** is an academic prototype for diabetes self-management education. This delivery implements **Phases 1–5 and the Phase 7 food-image workflow**: a structured synthetic dataset, a SQLite data layer, patient profile/context helpers, analytics, source-attributed medical retrieval, a consent-gated grounded AI assistant, and user-confirmed meal logging from food photographs.

> **Academic prototype / educational self-management assistant. Not a clinical decision-support system.** All included patient, meal, glucose, and lifestyle records are fictional synthetic demo data. The glucose values are not measurements or medically validated predictions. Food nutrition values are illustrative estimates, not validated against an authoritative food-composition table. Do not use this prototype or its numbers to make medical or dietary treatment decisions. Consult a qualified healthcare professional.

## Phase 1–5 contents

- Ten fictional Type 2 diabetes demo profiles.
- Seven days of related records per patient: 210 meals, 140 glucose readings, and 70 lifestyle records.
- A 34-item Indian/Kerala food reference table. Nutrition numbers are explicitly marked as unverified demo estimates.
- A Streamlit dashboard with patient selection, profile, meal history, glucose charts and summary statistics, lifestyle charts, and the food reference table.
- A deterministic data generator that refreshes the seven-day window to include the current date.
- A SQLite database initialized from the CSV files, with validated, transactional imports and change detection.
- Patient profile, descriptive summary, configurable recent-history, nutrition lookup, and structured context builder functions.
- Meal, glucose, lifestyle, and combined patient analytics with date-range filtering, data-quality reporting, and future-AI context preparation.
- An interactive Analytics dashboard with meal, glucose, and lifestyle visualizations.
- A local medical knowledge base, source metadata registry, persisted ChromaDB index, and evidence-only search page.
- An OpenAI-compatible LLM provider abstraction and a patient-aware assistant that uses selected-period analytics and local retrieved evidence.
- A Streamlit chat page with explicit consent before sending any question, retrieved passages, recent chat history, or relevant patient context to the configured LLM provider.
- Deterministic request/response safeguards, emergency escalation, verified source display, and bounded session-only conversation memory.

Phase 7 adds a **Food Image Analysis** page. Meal Recommendations remains a later-phase placeholder; no Phase 6 recommendation engine exists in this repository.

## Project structure

```text
.
├── app.py
├── requirements.txt
├── .env.example
├── data/
│   ├── generate_demo_data.py
│   ├── patients.csv
│   ├── meals.csv
│   ├── glucose.csv
│   ├── lifestyle.csv
│   └── food_nutrition.csv
├── knowledge_base/
│   ├── diabetes/
│   ├── lifestyle/
│   ├── metadata/
│   └── nutrition/
├── src/
│   ├── llm/
│   ├── chatbot/
│   ├── food_vision/
│   ├── database/
│   │   └── database.py
│   ├── patient_profile/
│   │   ├── context.py
│   │   └── profile.py
│   ├── analytics/
│   │   ├── meal_analytics.py
│   │   ├── glucose_analytics.py
│   │   ├── lifestyle_analytics.py
│   │   └── patient_analytics.py
│   ├── glucose_analysis/
│   ├── rag/
│   │   ├── document_loader.py
│   │   ├── document_cleaner.py
│   │   ├── chunker.py
│   │   ├── embeddings.py
│   │   ├── vector_store.py
│   │   ├── retriever.py
│   │   └── pipeline.py
│   ├── recommendation/
│   └── safety/
├── frontend/
├── tests/
└── vector_db/
```

## Install and run locally (Windows PowerShell)

Python 3.11 or newer is recommended.

```powershell
py -3.11 -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
pip install -r requirements.txt
python data\generate_demo_data.py
streamlit run app.py
```

If script activation is restricted by your PowerShell policy, run the virtual environment's executables directly:

```powershell
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
.\.venv\Scripts\python.exe data\generate_demo_data.py
.\.venv\Scripts\streamlit.exe run app.py
```

Streamlit prints a local URL (normally `http://localhost:8501`). Copy `.env.example` to `.env` to configure local paths and LLM settings; the app loads it at startup. The default local embeddings need no API key. Never put real credentials in source control.

## Dataset schemas

- `patients.csv`: patient ID, age, gender, diabetes type, diet and cuisine preferences, activity, restrictions, goals, and lifestyle notes.
- `meals.csv`: patient/date/meal, food items, portions, illustrative nutrition estimates, and `is_demo`.
- `glucose.csv`: patient/date/time, value in mg/dL, context, and `is_demo`.
- `lifestyle.csv`: patient/date, sleep, exercise, water, stress, and `is_demo`.
- `food_nutrition.csv`: food identifiers, regional names, category, serving size, illustrative nutrition estimates, source note, and `is_demo`.

The generator links every meal, glucose, and lifestyle record to one of the ten profile IDs. All generated records carry an `is_demo` flag; each food value also includes an explicit synthetic/unverified source note. Run the generator again to replace the CSVs with a fresh rolling seven-day window.

## Phase 2: SQLite and patient context

The CSV files remain the source/demo data. On first startup, GlucoCare validates the CSV schemas, IDs, dates, numeric values, and patient references, then imports the records into `data/glucocare.sqlite3`. The importer stores CSV SHA-256 digests and skips the import when the files have not changed. A changed source set is validated and imported transactionally; an invalid import leaves the previous database records intact.

The SQLite database contains:

- `patients`: profile, preferences, restrictions, goals, lifestyle notes, and demo label.
- `meals`: meal type, combined food/portion description, and meal nutrition estimates.
- `glucose`: dated and timed reading values, context, and demo label.
- `lifestyle`: sleep, exercise, water, stress, and demo label.
- `food_nutrition`: food reference and illustrative nutrition values/source.
- `import_metadata`: source CSV hashes and import timestamps.

Patient histories and nutrition lookups are indexed/looked up in SQLite. Profile and history data are also exposed as programmatic helpers:

```python
from src.patient_profile.context import build_patient_context
from src.patient_profile.profile import generate_patient_summary, get_patient_profile

context = build_patient_context("P001")
summary = generate_patient_summary("P001")
profile = get_patient_profile("P001", days=7)
```

These functions return `None` for an unknown patient; missing histories are empty lists and missing summary averages are `None`. `build_patient_context` includes the profile, diabetes and diet information, restrictions, goals, recent meals/glucose/lifestyle, food preferences, and a descriptive summary. It does not make medical interpretations.

Initialize or update the SQLite database from the CSVs manually:

```powershell
python -m src.database.database
```

The Streamlit app initializes the database on startup. If you edit CSV files while it is running, use **Reload CSV data into database** in the sidebar. The database file is local and ignored by Git.

The dashboard patient selector updates all sections. Meal history can be filtered by date and meal type; glucose history by date and context; lifestyle history by date. Empty history is displayed as missing rather than filled with fabricated values. All displayed values remain labeled as synthetic demo data.

## Phase 3: analytics

Analytics are separated from the Streamlit UI in `src/analytics/`:

- `meal_analytics.py`: meal counts and type distribution, food frequencies, daily sums from recorded meals, averages across recorded meals/recorded days, and optional meal-time/gap summaries.
- `glucose_analytics.py`: reading count, average/minimum/maximum/median/standard deviation, trends, daily statistics, context averages, and optional time-windowed meal/glucose associations.
- `lifestyle_analytics.py`: record counts and descriptive sleep, exercise, water, and ordinal stress summaries.
- `patient_analytics.py`: combined patient analytics, data availability/quality counts, and `get_ai_context(...)` for future modules. It does not invoke an LLM.

Analytics functions accept optional inclusive `start_date` and `end_date` values (`date`, `datetime`, or `YYYY-MM-DD` strings). If omitted, they use the last seven days; `days` can change the default window. For example:

```python
from src.analytics.patient_analytics import build_patient_analytics, get_ai_context
from src.analytics.glucose_analytics import get_glucose_summary

summary = get_glucose_summary("P001", "2026-10-09", "2026-10-15")
analytics = build_patient_analytics("P001", days=14)
context = get_ai_context("P001", start_date="2026-10-09", end_date="2026-10-15")
```

The Analytics page offers last 3, 7, 14, or 30 days and a custom date range. It shows descriptive meal frequencies and recorded nutrition, glucose trends/context/daily averages, lifestyle metrics, exploratory meal-glucose associations, and a data-quality section. Glucose and meal charts do not use medical thresholds or make causal claims.

Meal timestamps are optional because the Phase 1 demo CSV does not contain them. SQLite initialization adds a nullable `meal_time` column to older databases and imports it only if it is present in `meals.csv`; timing and meal-glucose analysis are skipped if usable times are absent. The association helper only compares readings in a configurable time window around explicitly timestamped meals and labels the result as exploratory, not causal.

Nutrition sums represent only the meals recorded in the data, not necessarily a complete day's intake. The dashboard labels these as sums from recorded meals. Stress scores are a descriptive ordinal mapping (Low=1, Moderate=2, High=3), not a clinical scale.

## Verification

Run the focused dataset and dashboard tests with:

```powershell
python -m unittest discover -s tests -v
```

The tests validate the Phase 1 dataset, Phase 2 database schema/import, patient queries, recent-history windows, profile/context builder, analytics calculations/date filters, empty/invalid histories, and timestamp-dependent associations. They do not evaluate clinical accuracy or model behavior.

## Phase 4: Medical Knowledge Base + RAG

The retrieval-only subsystem searches public reference excerpts and returns source-linked passages. It does **not** generate medical answers, apply patient context, diagnose, or provide clinical advice.

### Sources and provenance

Source records live in [knowledge_base/metadata/sources.json](knowledge_base/metadata/sources.json). Every source document entry records its source organization, title, canonical URL, topic, document type, access date, and a note about reuse. The initial collection contains short, attributed excerpts from official NIDDK and WHO diabetes pages. These pages were reachable at implementation time. Only limited excerpts are bundled; confirm each source's current reuse terms before redistribution, and consult the source URL for the complete and current material. No random blogs, patient records, or generated medical source content are included. If source metadata is absent, the UI reports it as unavailable rather than inventing a citation.

Supported local source formats are PDF, TXT, Markdown, and MD. Put public/reference materials under `knowledge_base/diabetes/`, `knowledge_base/nutrition/`, or `knowledge_base/lifestyle/` and add their provenance to `knowledge_base/metadata/sources.json`. The knowledge base must never contain patient information, credentials, or secrets.

### Chunking, embeddings, and vector database

The loader extracts PDF pages and reads UTF-8 text/Markdown. Conservative text cleaning normalizes whitespace and common extraction artifacts without rewriting source content. Chunking prefers paragraph and sentence boundaries, uses configurable character-size limits and overlap, and carries source metadata into every chunk.

The default embedding is deterministic local feature hashing (`GLUCOCARE_EMBEDDING_PROVIDER=hashing`), which requires no API key or model download. It is lightweight and less semantically capable than a trained embedding model. Retrieval reranks the vector results using query-term coverage and topic/title matches to improve topical relevance. An optional `sentence-transformers` provider can use a locally installed model by setting `GLUCOCARE_EMBEDDING_PROVIDER=sentence-transformers` and `GLUCOCARE_EMBEDDING_MODEL` to that model's identifier/path; install `sentence-transformers` separately and ensure the model is available locally if running offline. No external embedding API is called.

ChromaDB persists the index under `vector_db/` by default. At initialization, source files are loaded and chunked to compute a corpus fingerprint. Embeddings and the index are rebuilt only if the corpus/configured model changes, the index is missing, or the user explicitly requests a rebuild. The local index is ignored by Git.

Configuration variables (copy `.env.example` to `.env` if desired; environment loading is handled by the host shell/deployment):

| Variable | Default | Purpose |
| --- | --- | --- |
| `GLUCOCARE_KNOWLEDGE_BASE` | `knowledge_base` | Source directory |
| `GLUCOCARE_VECTOR_DB` | `vector_db` | Chroma persistence directory |
| `GLUCOCARE_EMBEDDING_PROVIDER` | `hashing` | `hashing` or optional `sentence-transformers` |
| `GLUCOCARE_EMBEDDING_MODEL` | provider-specific | Embedding model identifier |
| `GLUCOCARE_CHUNK_SIZE` / `GLUCOCARE_CHUNK_OVERLAP` | `800` / `100` | Retrieval text chunk configuration |
| `GLUCOCARE_TOP_K` | `5` | Default number of retrieved passages |
| `GLUCOCARE_MIN_RELEVANCE_SCORE` | `0.08` | Minimum hybrid retrieval ranking score |

### Build and search

Build/rebuild from the repository root:

```powershell
python -c "from src.rag.pipeline import initialize_rag; print(initialize_rag(force_rebuild=True))"
```

Run one retrieval programmatically:

```powershell
python -c "from src.rag.pipeline import retrieve_context; print(retrieve_context('What are carbohydrates?', top_k=5))"
```

Open Streamlit and choose **Medical Knowledge** to enter a question, set Top-K, and inspect evidence/source attribution. The page displays source, document title, topic, type, relevance score, original URL, and retrieved passage. Relevance scores are hybrid rankings (local vector similarity plus topic/title/term matches); they do not measure medical correctness. An empty or unrelated query does not produce a generated answer.

### Retrieval evaluation

`tests/rag_queries.json` contains 20 representative questions labeled with expected topics, not asserted medical answers. Run:

```powershell
python -m src.rag.evaluate
```

The evaluation measures topic-label Hit Rate@K against the current local source corpus. With the bundled baseline (6 documents, 6 chunks, 2 sources), the 20-query set measured 20/20 topic hits at K=5 (100%). This is a small retrieval smoke/evaluation set, not clinical validation, factual answer scoring, or proof that retrieval results are sufficient. Re-run it after changing the source corpus or retrieval configuration.

## Phase 5: LLM + Grounded AI Assistant

### Architecture and grounding

The **AI Assistant** page calls `src/chatbot/chatbot.py`. It reuses the selected patient's existing `get_ai_context` analytics and date-filtered database history, then retrieves relevant excerpts through the Phase 4 RAG pipeline. Context is minimized by question type, bounded in size, excludes the internal patient ID, and uses explicit delimiters. General medical claims must have retrieved evidence; unsupported medical questions receive a no-evidence response. Patient-record observations and general medical guidance are kept separate. Citations are attached from retrieved source metadata by the application, not generated by the LLM.

Retrieved excerpts, patient records, and conversation history are untrusted data, never prompt instructions. Deterministic safeguards intercept medication-change and prompt-injection requests, provide emergency escalation for recognized urgent-symptom phrases, and reject several unsafe response patterns or unsupported measured values. This is a basic safety layer, not comprehensive clinical risk detection. The assistant must not diagnose, prescribe, infer causality, or replace professional advice.

The provider uses the OpenAI-compatible Chat Completions API behind `src/llm/client.py` and `src/llm/provider.py`. The default API URL/model are configurable, so compatible hosted or local services can be selected without changing chatbot code. No LLM package was added; the API client uses Python's standard library.

| Variable | Default | Purpose |
| --- | --- | --- |
| `LLM_API_KEY` | empty | Secret API credential; required to generate responses |
| `LLM_MODEL` | `gpt-4o-mini` | Provider model identifier |
| `LLM_BASE_URL` | `https://api.openai.com/v1` | HTTPS OpenAI-compatible API base (localhost HTTP allowed for local development) |
| `LLM_TEMPERATURE` | `0.2` | Generation temperature, 0–2 |
| `LLM_MAX_TOKENS` | `700` | Maximum generated tokens |
| `LLM_TIMEOUT_SECONDS` | `30` | API request timeout |
| `MAX_HISTORY_MESSAGES` | `10` | Maximum recent user/assistant messages in session memory |
| `RAG_TOP_K` | `5` | Retrieved passages supplied to the assistant |

### Privacy and run steps

1. Install dependencies using the Windows instructions above.
2. Run `Copy-Item .env.example .env`, then edit the local `.env` file (for example, `notepad .env`). Set `LLM_API_KEY` and the desired `LLM_MODEL`; set `LLM_BASE_URL` if the provider uses a compatible endpoint. `.env` is ignored by Git.
3. Start the app with `streamlit run app.py`.
4. Select a demo patient in the sidebar, choose **🤖 AI Assistant**, and set the analysis period.
5. Review and check the consent box. The question, the minimum relevant selected-patient context, retrieved passages, and recent session conversation are sent to the configured provider. Without consent, chat input is disabled. Do not use real personal health information unless your provider, deployment, and applicable privacy requirements have been reviewed.
6. Ask a question. Retrieved source excerpts and a concise indicator of the patient context used are expandable below the response. Conversation state is held only in the current Streamlit session, is bounded by `MAX_HISTORY_MESSAGES`, and is not saved to the database.

Missing API keys show a setup message; provider failures, retrieval failures, empty questions, and invalid responses are surfaced without revealing credentials. A temporary provider rate limit may clear after waiting; if the provider reports insufficient quota, check the API account's billing and usage limits. Application logs include an anonymized patient identifier, question category, retrieval count/status, LLM status, and safety result; they do not include full questions, patient histories, conversation contents, or API keys.

### Example interaction (illustrative, not a captured model result)

**User:** Can I include puttu in my breakfast?

**GlucoCare:** Puttu need not be labeled universally good or bad. Your recorded preferences and the available source guidance can help frame portion and overall meal-composition considerations, but this prototype cannot make a personal treatment decision. Discuss individualized meal planning with your healthcare professional.

**Sources:** Only citations returned by the local retrieval system are attached. If it finds no relevant evidence, the assistant says so instead of fabricating a source.

### Tests and evaluation

Run deterministic tests (all API calls are mocked; no real key or external request is needed):

```powershell
python -m unittest tests.test_llm_client tests.test_prompt_builder tests.test_chatbot tests.test_safety tests.test_grounding -v
python -m unittest discover -s tests -v
```

`tests/llm_evaluation.json` contains 24 human-review questions spanning general diabetes, nutrition, Indian/Kerala foods, patient records, glucose, lifestyle, safety, unsupported questions, and prompt injection. It defines expected behaviors, not model answers or scores. Review actual responses for groundedness, citation correctness, personalization, hallucination resistance, safety, and regional relevance when evaluating a configured provider. No LLM quality score or provider comparison is claimed here; results depend on the model and source corpus.

### Limitations

- The API requires an internet-accessible OpenAI-compatible provider and a valid user-supplied key; no provider call was made during development validation.
- The bundled local medical corpus is small and retrieval relevance is limited. The RAG feature-hashing embeddings are not a trained semantic model.
- The built-in patient and nutrition data are synthetic demo values; no real clinical records or validated food-composition data are included.
- Prompt instructions and deterministic phrase checks cannot guarantee safe or clinically correct output. Emergency phrase detection is basic and must not be used as a triage system.
- The assistant has not been clinically validated, and there is no generic-LLM baseline comparison.

## Phase 7: Food Image Analysis

The **Food Image Analysis** page uses the existing sidebar patient selection. Upload a JPEG, PNG, or WEBP photo (default limit 10 MB), inspect the sanitized preview, analyze it with a vision-capable OpenAI-compatible model, then review/correct each label, explicitly select a food-record match (or mark it unmatched), state servings in the selected database record's own serving unit, set meal date/time/type, and press **Confirm and Save Meal**. No meal is persisted during upload or analysis. A confirmed record appears in the existing meal history and analytics. Changing patient clears the in-session analysis. Uploaded image bytes and image metadata are not saved to the database; the image is re-encoded in memory as JPEG, stripping metadata, before provider submission.

Food identification, alternatives, and optional visual portion estimates are uncertain. The model-reported confidence is not calibrated; the model's weight estimate is informational only and is never silently converted into servings. A user explicitly supplies quantity as a count of the selected database record's stated serving; no gram/millilitre/household-unit conversion is assumed.

The bundled food table contains synthetic demo estimates marked `is_demo`, not verified nutrition values. Those records can be offered as match candidates but are never used to calculate meal nutrition. The calculation helpers accept nutrition only from non-demo records with source provenance. Currently, no verified food composition records are bundled, so nutrition totals for the built-in data will be shown as unavailable. Unmatched items can still be logged with nutrition stored as SQL `NULL`, not fabricated zeroes. Existing databases are migrated to allow nullable meal nutrient fields while preserving historical rows. Daily meal analytics show a nutrient as unavailable if any meal on that day lacks that value.

### Vision provider and configuration

The vision client sends the normalized image to an OpenAI-compatible **vision-capable** Chat Completions endpoint. The default is `gpt-4o-mini` at `https://api.openai.com/v1`. A text-only model is not sufficient. Configure these values in the local `.env` (start from `.env.example`):

| Variable | Default | Purpose |
| --- | --- | --- |
| `VISION_API_KEY` | empty | Provider key; if empty, `LLM_API_KEY` is used |
| `VISION_MODEL` | `gpt-4o-mini` | Vision-capable model name |
| `VISION_BASE_URL` | `https://api.openai.com/v1` | OpenAI-compatible API base URL; local HTTP endpoints are allowed |
| `VISION_TIMEOUT_SECONDS` | `45` | Provider request timeout |
| `FOOD_IMAGE_MAX_UPLOAD_MB` | `10` | Upload limit, configurable from 1 to 50 MB |

The API provider receives the image bytes for classification. Do not upload identifiable or sensitive photographs unless your deployment/provider privacy terms have been reviewed. The app does not retain those bytes in meal history or application-level session state after analysis, but Streamlit/browser upload handling and the external provider are outside this prototype's storage controls.

Pillow is installed with `pip install -r requirements.txt`. No real API key is required for tests. Tests use generated in-memory images and mocked vision clients; no licensed photographic evaluation dataset or model-performance benchmark is bundled, and no recognition accuracy is claimed.

### Food-image safety and limitations

- The image is decoded and checked against both extension and actual JPEG/PNG/WEBP format, limited by file size and pixel dimensions, EXIF-oriented, resized to at most 2048×2048, and re-encoded without metadata.
- The constrained prompt treats image-associated content as data, not instructions. The model cannot establish exact recipes, hidden ingredients, allergens, portion weights, nutrition, diabetes status, or glucose response from appearance.
- Model responses are schema-validated. Invalid structured responses receive at most one retry; timeouts, provider failures, configuration issues, and invalid responses produce user-facing errors without exposing credentials.
- Food matches are suggestions only. Nothing is selected automatically, and similar dishes are not silently substituted.
- The required disclaimer is shown on the page. This feature is an educational prototype, not a diagnostic or clinical decision-support system.

Run the feature from the same project root and launch command as the rest of the app:

```powershell
streamlit run app.py
```

Choose **Food Image Analysis** in the sidebar after selecting a demo patient. If no vision key/model is configured, image analysis displays a setup error; other pages continue to use their existing behavior.

Targeted food-image and database tests:

```powershell
python -m unittest tests.test_food_vision tests.test_food_vision_database -v
```

The existing **Meal Recommendations (Coming later)** entry is not replaced or modified by this phase. Food-image logs are visible through existing meal-history and analytics readers; there is no Phase 6 recommendation implementation to integrate with yet.
