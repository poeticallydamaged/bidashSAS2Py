# SAS Viya Visual Analytics & End-to-End Home Equity BI Dashboard (Python / Streamlit + Plotly)

A self-contained, production-grade Python replication of the **SAS Viya for Learners End-to-End Financial Risk Use Case** documented in `guide.pdf` using `home_equity.csv` (`data/dataset.csv`, `5,960` home equity loan applications across `18` source columns).

---

## 1. Project Structure

```text
bi_dashboard/
├── .streamlit/
│   └── config.toml                 # SAS Viya Opal/Marine color theme, typography & server config
├── data/
│   ├── dataset.csv                 # Primary input dataset (5,960 rows x 18 columns)
│   └── home_equity.csv             # Alias copy matching the SAS Viya tutorial filename
├── src/
│   ├── __init__.py
│   ├── data_engine.py              # Cached ingestion, SAS date parsing, LTV calculation, NLG & ML engine
│   ├── visuals.py                  # Plotly figure generators matching SAS Viya Visual Analytics objects
│   └── components.py               # KPI metric cards, Model Card health badges, CSS & filter controls
├── app.py                          # Main wide-layout multi-tab Streamlit application entry point
├── requirements.txt                # Pinned Python dependencies
├── README.md                       # Architecture, SAS-to-Python mapping & run instructions
└── run.sh                          # One-click virtual environment setup & execution script
```

---

## 2. SAS Viya to Python Translation Mapping

Every construct from the 6 tasks in `guide.pdf` is translated directly into modular Python equivalents:

| SAS Viya Construct (`guide.pdf`) | Python / Streamlit + Plotly Implementation |
| :--- | :--- |
| **Task 1: SAS Studio Data Prep (`Import_and_load_HOMEEQUITY1.sas`, p. 14–16)** | `load_and_prep_data()` in `src/data_engine.py`: maps all 18 human-readable variable labels, converts `CITY` to Proper Case (`str.title()`), computes **Loan to Value ratio** (`LTV = LOAN / VALUE`), and parses SAS numeric `APPDATE` (days since `1960-01-01`) into `MMMYYYY` (`MONYY7`, `JAN2017`–`DEC2021`). |
| **Task 2: Data Roles & Default Aggregations (p. 17–19)** | Converts two-level numeric `BAD` (`0` vs `1`) to discrete category `Loan Status`, and changes default aggregation of all 11 continuous measures (`CLAGE`, `MORTDUE`, `LOAN`, `DEBTINC`, `LTV`, `CLNO`, `DELINQ`, `DEROG`, `NINQ`, `VALUE`, `YOJ`) from `Sum` to `Average` (`mean`). |
| **Task 2: `Interactive charts` 2×2 Grid & Cross-Filtering (p. 20–25)** | **Top-Left:** Donut Chart (*Frequency of Loan Status*) with center count (`6K`) and outside percentage labels in **16pt Bold Deep Purple (`#4B1E78`)**. **Top-Right:** Histogram (*Frequency of Amount of Loan Request*). **Bottom-Left:** 11×11 Pearson Correlation Matrix (*Correlation of Selected Measures*) with outlier flags (`CLAGE`, `MORTDUE`, `LOAN`, `DEBTINC`, `LTV`). **Bottom-Right:** Monthly Time Series (*Frequency of Loan Application Date*). Linked cross-filtering updates all visuals simultaneously. |
| **Task 2 & Workbook Extensions: Dual-Axis Bar-Line & Geo Hierarchy (p. 16, 55)** | Grouped Bar + Line Chart (*Amount Due on Existing Mortgage, Value of Current Property by Job Category*, Fig. 23 p. 16) plus interactive US State Choropleth and *Region → Division → State* drill-down Treemap. |
| **Task 3: `Explain Loan Status` Automated Explanation (p. 25–28)** | `compute_automated_explanation()` & `generate_factor_nlg_summary()`: generates the NLG banner (*"1 is less common at 19.95% (1.2K of 6K)... The three most related factors are Debt to Income Ratio, Number of Delinquent Credit Lines, and Value of Current Property"*), normalized Relative Importance chart (`DEBTINC` = 1.00, `DELINQ` = 0.35), stacked relationship histograms, and the 4 Maximized View diagnostic tabs (`Explanation Description`, `Screening Results`, `Relative Importance`, `High / Low Groups & Anomalies`). |
| **Task 4: Model Studio Pipelines & Fairness on `JOB` (p. 28–42)** | `train_and_evaluate_sas_models()`: trains **Gradient Boosting (Champion)**, **Logistic Regression**, and **Open Source Python (Scikit-Learn Random Forest)** across stratified Train (60%) / Validate (30%) / Test (10%) partitions, reporting KS, Accuracy, ASE, ROC curves, Cumulative Lift, and Sensitive Attribute `JOB` Fairness & Bias metrics. |
| **Task 5: SAS Model Manager Model Card (p. 43–45)** | `render_model_card_health_row()`: renders the Model Health Scorecard badges (**Accuracy: PASS**, **Generalizability: PASS**, **Fairness: FAIL** due to `JOB` prediction bias disparity across `Sales` vs `Office`/`ProfExe`). |
| **Task 6: Intelligent Decisioning `Loan Decision (Final)` Flow (p. 45–54)** | `evaluate_decision_flow_single()` & `evaluate_batch_decision_flow()`: replicates `Start` → `Gradient Boosting` → `P_BAD1 < 0.20` branch → `Check Debt to Income (DEBTINC < 40)` / `Declined Applications` → Python node `Create Email Message (Final)` → `End`, with both an interactive applicant simulator and a portfolio-wide Sankey diagram. |

---

## 3. Quick Start & Execution Instructions

### Option A: One-Click Launch via `run.sh` (macOS / Linux)
```bash
chmod +x run.sh
./run.sh
```

### Option B: Manual Virtual Environment Setup
```bash
# 1. Create and activate a virtual environment
python3 -m venv .venv
source .venv/bin/activate        # On Windows: .venv\Scripts\activate

# 2. Install dependencies
pip install -r requirements.txt

# 3. Launch the Streamlit application
streamlit run app.py
```

The dashboard will open automatically in your browser at `http://localhost:8501`.
