# Enterprise Credit Risk Analytics & AI Platform

A Basel-aligned credit risk scoring, portfolio diagnostics, anomaly detection, and natural language analytics engine built on bank portfolio data (14,685 loan accounts).

---

## 1. Technical Environment & Python Version
- **Python Version**: `Python 3.11+` (tested and developed on `Python 3.13`)

---

## 2. Quickstart Instructions (Run in 3 Commands)

Execute the following commands from the `complete_assignment/` directory:

```bash
# 1. Install dependencies
pip install -r requirements.txt

# 2. Build dataset, train champion scorecard, and detect anomalies
python -c "from src.data_pipeline import build_master_dataset; from src.model import train_and_score_portfolio; from src.anomaly_detector import detect_portfolio_anomalies; import os; os.makedirs('data/processed', exist_ok=True); df = build_master_dataset('data/raw'); scored_df, _, _, metrics, _ = train_and_score_portfolio(df); final_df = detect_portfolio_anomalies(scored_df); final_df.to_csv('data/processed/processed_loan_portfolio.csv', index=False); print(f'Pipeline complete. Scored {len(final_df):,} accounts. Model ROC-AUC: {metrics[\"ROC-AUC\"]:.4f}, KS: {metrics[\"KS_pct\"]:.2f}%')"

# 3. Start the dashboard
streamlit run app/app.py
```

---

## 3. LLM API Key Configuration & Offline Fallback

The application features a provider-based LLM architecture (`src/llm_provider.py`) powering the grounded **Ask the Portfolio** engine.

### How to set the LLM API Key:
Copy `.env.example` (or edit `.env`) in the root of `complete_assignment`:

#### Option A: LangChain `ChatOpenAI` (Recommended)
```env
LLM_PROVIDER=langchain_openai
LANGCHAIN_OPENAI_API_KEY=your_api_key_here
LANGCHAIN_OPENAI_BASE_URL=https://your-custom-endpoint.com/v1   # Optional custom gateway/proxy
LANGCHAIN_OPENAI_MODEL_NAME=gpt-4.1
```

#### Option B: Direct OpenAI SDK
```env
LLM_PROVIDER=openai
OPENAI_API_KEY=your_api_key_here
OPENAI_BASE_URL=                                              # Optional
OPENAI_MODEL_NAME=gpt-4o-mini
```

### What happens if no key is set?
- **Zero Crashes / Full Functionality**: The system automatically detects missing or template keys and gracefully routes to **Deterministic Offline Mode**.
- In offline mode, the engine executes exact pandas aggregation queries over the portfolio DataFrame and formats the answer using a grounded deterministic template with full source attribution (`loan_type`, `target_default`, etc.).
- Financial calculations never hallucinate regardless of API availability.

---

## 4. Repository Mapping: Where Each Brief Item Lives

| Deliverable / Brief Requirement | File / Directory Location | Description |
| :--- | :--- | :--- |
| **Exploratory Analysis & Audit** | `test/final_solution.ipynb` | Comprehensive 19-cell sequential notebook covering data distribution audits, skewness tests, categorical cleaning, and cohort imputation. |
| **Data Ingestion & Cleaning** | `src/data_pipeline.py` | ETL pipeline handling gender code harmonization, median imputations, and one-to-many aggregations to prevent Cartesian explosion (1 row per `account_id`). |
| **Predictive Model & Basel Scorecard** | `src/model.py` | Production champion Logistic Regression (`class_weight='balanced'`), ROC-AUC (0.6727), Gini (0.3455), KS-statistic (26.13%), calibration, and risk tiers (Low, Medium, High). |
| **Local Risk Drivers / Explainability** | `src/model.py` (`get_account_top_drivers`) | Calculates feature contributions for individual accounts based on standardized coefficients. |
| **AI Feature 1: Anomaly Detection** | `src/anomaly_detector.py` | Transparent rule-based anomaly detector (Delinquency spikes, cohort payment shortfalls > 2.5 SD, bureau divergence) with operational review governance. |
| **AI Feature 2: Ask the Portfolio** | `src/ask_portfolio.py` & `src/llm_provider.py` | Multi-provider grounded Q&A engine executing Python calculations before passing context to LangChain/OpenAI or offline templates. |
| **Interactive Dashboard (4 Views)** | `app/app.py` | Streamlit application with: <br>1. *Portfolio View* (macro metrics, slice-and-dice)<br>2. *Account Investigation* (scorecard, gauge, drivers)<br>3. *Ask the Portfolio* (grounded Q&A)<br>4. *Anomaly Review Queue* (governance triage). |
| **Raw & Processed Data** | `data/raw/` & `data/processed/` | Houses raw input CSVs and the reproducible processed master dataset. |
| **Executive & Architecture Docs** | `docs/` | Destination folder for `architecture.pdf` and `solution_overview.pdf`. |
