# GlucoCare

**GlucoCare: A Personalized Multimodal Generative AI Assistant for Diabetes Care and Regional Diet Management**

GlucoCare is an academic prototype designed to support diabetes self-management education using AI, patient data analytics, medical knowledge retrieval, and food image analysis.

> **Disclaimer:** This is an educational prototype, not a clinical decision-support system. Patient records are synthetic, and included food nutrition values are illustrative, unverified estimates. It must not replace professional medical advice.

## Features

- **Patient Dashboard:** View synthetic patient profiles, meal history, glucose readings, and lifestyle records.
- **Patient Context Management:** Organize patient preferences, dietary restrictions, goals, and recent history.
- **Analytics:** Summarize meal, glucose, and lifestyle data.
- **Medical Knowledge Retrieval (RAG):** Retrieve relevant passages from the included knowledge base with source attribution.
- **AI Assistant:** Generate responses using retrieved evidence and relevant patient context, with basic safety safeguards and user consent.
- **Food Image Analysis:** Upload a food photo, review suggested labels, match foods to database records, specify servings, and confirm meal logging.

## Project Status

- Phases 1–5: Implemented in the current codebase.
- Phase 6: Personalized meal recommendation engine is not yet implemented.
- Phase 7: Food image analysis workflow is present.
- Integration, testing, evaluation, and final deployment: Require verification and/or completion.

The presence of code does not guarantee that every feature has passed testing.

## Technology Stack

- Python
- Streamlit
- SQLite
- Pandas
- ChromaDB
- Retrieval-Augmented Generation (RAG)
- OpenAI-compatible LLM and vision APIs

## Project Structure

```text
GlucoCare/
├── app.py
├── requirements.txt
├── .env.example
├── data/
├── knowledge_base/
├── src/
│   ├── database/
│   ├── patient_profile/
│   ├── analytics/
│   ├── rag/
│   ├── llm/
│   ├── chatbot/
│   ├── food_vision/
│   └── recommendation/
├── frontend/
└── tests/
```

## Setup and Run

Python 3.11 or newer is recommended.

1. Clone or download this repository.
2. Create and activate a virtual environment:

   ```powershell
   py -3.11 -m venv .venv
   .\.venv\Scripts\Activate.ps1
   ```

3. Install dependencies:

   ```powershell
   python -m pip install --upgrade pip
   pip install -r requirements.txt
   ```

4. Generate synthetic demo data:

   ```powershell
   python data\generate_demo_data.py
   ```

5. Copy `.env.example` to `.env` and configure the required settings. Add your own API key if you want to use the external AI features.

6. Start the application:

   ```powershell
   streamlit run app.py
   ```

## Run Tests

From the project root:

```powershell
python -m unittest discover -s tests -v
```

Record the actual test results before claiming that the project passes.

## Safety and Limitations

- All bundled patient records are fictional.
- Food nutrition values are demo estimates, not validated clinical nutrition data.
- AI-generated responses and image-based food identification may be inaccurate.
- No clinical validation or food-image recognition accuracy benchmark is claimed.
- Do not upload API keys, `.env` files, or private patient information to GitHub.

## Team Development

The existing codebase is the starting point for further development. The next major feature is the personalized meal recommendation engine, followed by integration testing, evaluation, documentation, and deployment.
