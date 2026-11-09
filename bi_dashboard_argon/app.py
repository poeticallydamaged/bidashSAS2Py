"""Main Streamlit entry point replicating the SAS Viya Home Equity End-to-End Dashboard.

Pages / Tabs replicated from `guide.pdf`:
- Tab 1: **Interactive Charts (Task 2 — SAS Visual Analytics Page 1)**
  - 2x2 Visual Grid with Linked Cross-Filtering ("Automatic actions on all objects"):
    1. Donut Chart (`Frequency of Loan Status`, 16pt Bold Deep Purple labels, `6K` center count)
    2. Histogram (`Frequency of Amount of Loan Request`)
    3. Correlation Matrix (`Correlation of Selected Measures` across all 11 continuous items + outlier badges)
    4. Time Series Line Plot (`Frequency of Loan Application Date`, `MMMYYYY` format)
  - Plus Figure 23 (`Amount Due on Existing Mortgage, Value of Current Property by Job Category`)
    and Geographic Hierarchy Explorer (`Region -> Division -> State -> City`).
- Tab 2: **Explain Loan Status (Task 3 — SAS Visual Analytics Page 2 Automated Explanation)**
  - Target event switcher (`1` vs `0`), NLG summary banner, Relative Importance bar chart
    (`DEBTINC` = 1.00, `DELINQ` = 0.35, etc.), interactive factor relationship histogram/bar chart
    with NLG bullets, and the 4 Maximized View diagnostic tabs (`Explanation Description`,
    `Screening Results`, `Relative Importance`, `High / Low Groups & Anomalies`).
- Tab 3: **Model Studio, Fairness & Model Card (Tasks 4 & 5)**
  - Pipeline 1 (SAS AutoML: Gradient Boosting Champion vs Logistic Regression) & Pipeline 2
    (Open Source Python Scikit-Learn Random Forest) comparison table, ROC & Cumulative Lift curves,
    Sensitive Variable `JOB` Fairness & Bias tab, and SAS Model Manager **Model Card**
    (`Accuracy: Pass`, `Generalizability: Pass`, `Fairness: Fail`).
- Tab 4: **Loan Decision (Final) Flow & Data Prep (Tasks 1 & 6)**
  - Interactive SAS Intelligent Decisioning flow simulator (`Gradient Boosting` -> `P_BAD1 < 0.2`
    -> `Check Debt to Income < 40` -> `Create Email Message`), portfolio batch simulation Sankey,
    and Task 1 SAS Studio data preparation & variable dictionary inspector.
"""

from __future__ import annotations

import pandas as pd
import streamlit as st

from src.components import (
    inject_sas_viya_css,
    render_header_banner,
    render_kpi_row,
    render_model_card_health_row,
    render_sidebar_filters,
)
from src.data_engine import (
    CONTINUOUS_MEASURES,
    LABEL_TO_COL,
    SAS_LABELS,
    compute_automated_explanation,
    compute_correlation_matrix,
    compute_kpi_summary,
    compute_measure_summary_table,
    evaluate_batch_decision_flow,
    evaluate_decision_flow_single,
    filter_dataframe,
    generate_factor_nlg_summary,
    load_and_prep_data,
    train_and_evaluate_sas_models,
)
from src.visuals import (
    create_application_date_timeseries,
    create_correlation_matrix_heatmap,
    create_cumulative_lift_chart,
    create_decision_flow_sankey,
    create_factor_relationship_chart,
    create_job_fairness_chart,
    create_job_mortgage_value_chart,
    create_loan_request_histogram,
    create_loan_status_donut,
    create_regional_hierarchy_treemap,
    create_relative_importance_chart,
    create_roc_comparison_chart,
    create_us_state_choropleth,
)


def main() -> None:
    st.set_page_config(
        page_title="SAS Viya Home Equity Visual Analytics & Decisioning",
        page_icon="📊",
        layout="wide",
        initial_sidebar_state="expanded",
    )
    inject_sas_viya_css()
    render_header_banner()

    # Load and prepare dataset (Task 1)
    full_df = load_and_prep_data()

    # Render Global Sidebar Filters & Linked Cross-Filter Controls
    filters = render_sidebar_filters(full_df)
    filtered_df = filter_dataframe(
        full_df,
        loan_status_filter=filters["loan_status_filter"],
        regions=filters["regions"],
        divisions=filters["divisions"],
        states=filters["states"],
        jobs=filters["jobs"],
        reasons=filters["reasons"],
        loan_range=filters["loan_range"],
        date_range=filters["date_range"],
    )

    if filtered_df.empty:
        st.warning("No loan applications match the current filter combination. Please broaden the sidebar filters.")
        return

    # Top-level KPI Summary Row
    kpis = compute_kpi_summary(filtered_df, full_df)
    render_kpi_row(kpis)

    st.markdown("<div style='height:6px'></div>", unsafe_allow_html=True)

    # Multi-Tab Navigation Mirroring the SAS Viya Report Pages & End-to-End Lifecycle
    tab1, tab2, tab3, tab4 = st.tabs(
        [
            "📊 Page 1: Interactive Charts (Task 2)",
            "🧠 Page 2: Explain Loan Status (Task 3)",
            "⚖️ Page 3: Model Studio, Fairness & Model Card (Tasks 4 & 5)",
            "🔀 Page 4: Loan Decision Flow & SAS Data Prep (Tasks 1 & 6)",
        ]
    )

    # =========================================================================
    # TAB 1: PAGE 1 - INTERACTIVE CHARTS (TASK 2, GUIDE P. 17-25)
    # =========================================================================
    with tab1:
        ctrl_col1, ctrl_col2, ctrl_col3 = st.columns([2.2, 1.4, 1.4])
        with ctrl_col1:
            st.markdown(
                "**SAS Visual Analytics — Home Equity Analysis Report (`Interactive charts`)**  \n"
                "<span style='font-size:0.86rem;color:#5C6873;'>"
                "Automatic actions enabled: use the sidebar <b>Linked Cross-Filter: Loan Status</b> "
                "or quick-filter pills below to cross-filter the Histogram, Correlation Matrix, and Time Series."
                "</span>",
                unsafe_allow_html=True,
            )
        with ctrl_col2:
            color_hist_by_status = st.toggle(
                "Split Loan Request Histogram by Loan Status",
                value=(filters["active_status"] == "All"),
            )
        with ctrl_col3:
            show_signed_corr = st.toggle(
                "Show Signed Pearson r (-1 to +1) in Matrix",
                value=False,
                help="SAS Viya default correlation matrix displays absolute strength (Weak to Strong). Toggle to inspect signed +/- direction.",
            )

        # Row 1 of 2x2 SAS VA Grid: Donut Chart (Left) + Histogram (Right)
        r1_left, r1_right = st.columns(2)
        with r1_left:
            donut_fig = create_loan_status_donut(filtered_df, active_status=filters["active_status"])
            st.plotly_chart(donut_fig, use_container_width=True)
        with r1_right:
            hist_fig = create_loan_request_histogram(filtered_df, color_by_status=color_hist_by_status)
            st.plotly_chart(hist_fig, use_container_width=True)

        # Row 2 of 2x2 SAS VA Grid: Correlation Matrix (Left) + Application Date Line Chart (Right)
        r2_left, r2_right = st.columns(2)
        with r2_left:
            corr_raw, corr_abs = compute_correlation_matrix(filtered_df)
            corr_fig = create_correlation_matrix_heatmap(corr_raw, corr_abs, show_signed=show_signed_corr)
            st.plotly_chart(corr_fig, use_container_width=True)
            st.markdown(
                "<span style='font-size:0.82rem;color:#5C6873;font-weight:600;'>SAS VA Outlier Indicators (Task 2, p. 20): </span>"
                "<span class='outlier-badge'>⚠ Age of Oldest Credit Line (CLAGE)</span>"
                "<span class='outlier-badge'>⚠ Amount Due on Existing Mortgage (MORTDUE)</span>"
                "<span class='outlier-badge'>⚠ Amount of Loan Request (LOAN)</span>"
                "<span class='outlier-badge'>⚠ Debt to Income Ratio (DEBTINC)</span>"
                "<span class='outlier-badge'>⚠ Loan to Value ratio (LTV)</span>",
                unsafe_allow_html=True,
            )
        with r2_right:
            ts_fig = create_application_date_timeseries(filtered_df, show_default_rate=True)
            st.plotly_chart(ts_fig, use_container_width=True)

        st.markdown("---")
        # Additional SAS VA Report Views (Figure 23 p. 16 & Geographic Hierarchy p. 55)
        st.markdown("#### 🌎 Extended SAS Visual Analytics Views: Job Category Dual-Axis & Geographic Hierarchy")
        ext_col1, ext_col2 = st.columns(2)
        with ext_col1:
            job_bar_fig = create_job_mortgage_value_chart(filtered_df)
            st.plotly_chart(job_bar_fig, use_container_width=True)
        with ext_col2:
            geo_mode = st.radio(
                "Geographic Visualization Mode",
                options=[
                    "US State Choropleth — Default Rate (%)",
                    "US State Choropleth — Average Loan Request ($)",
                    "US State Choropleth — Average LTV (%)",
                    "Regional Hierarchy Treemap (Region → Division → State)",
                ],
                horizontal=True,
            )
            if "Treemap" in geo_mode:
                st.plotly_chart(create_regional_hierarchy_treemap(filtered_df), use_container_width=True)
            else:
                metric_name = geo_mode.split(" — ")[1]
                st.plotly_chart(create_us_state_choropleth(filtered_df, metric=metric_name), use_container_width=True)

    # =========================================================================
    # TAB 2: PAGE 2 - EXPLAIN LOAN STATUS (TASK 3, GUIDE P. 25-28)
    # =========================================================================
    with tab2:
        exp_top_left, exp_top_mid, exp_top_right = st.columns([1.3, 1.8, 1.9])
        with exp_top_left:
            target_event = st.selectbox(
                "Explain Target Event (Loan Status = )",
                options=["1", "0"],
                index=0,
                format_func=lambda x: f"Loan Status = {x} ({'Default / Delinquent' if x == '1' else 'Current / Paid'})",
            )
        explanation = compute_automated_explanation(filtered_df, target_event=target_event)
        imp_df = explanation["importance_df"]

        with exp_top_mid:
            selected_factor_label = st.selectbox(
                "Select Factor to Inspect Relationship (Right Panel)",
                options=imp_df["Factor"].tolist(),
                index=0,
            )
        selected_factor_col = LABEL_TO_COL.get(selected_factor_label, "DEBTINC")

        with exp_top_right:
            st.info(
                "💡 **SAS Viya Automated Explanation (Task 3, p. 25–28):** "
                "Screens 17 candidate predictors, removes redundant/temporal variables, and ranks relative importance."
            )

        # SAS Viya NLG Top Summary Banner (Task 3 p. 26)
        st.markdown(
            f"<div class='sas-nlg-banner'>{explanation['banner_text']}</div>",
            unsafe_allow_html=True,
        )

        # Side-by-Side Relative Importance Bar Chart & Relationship Plot
        exp_left, exp_right = st.columns([1.05, 1.15])
        with exp_left:
            rel_imp_fig = create_relative_importance_chart(imp_df, selected_factor=selected_factor_label)
            st.plotly_chart(rel_imp_fig, use_container_width=True)

        with exp_right:
            rel_plot_fig = create_factor_relationship_chart(filtered_df, factor_col=selected_factor_col)
            st.plotly_chart(rel_plot_fig, use_container_width=True)

            # Natural Language Generation (NLG) Factor Footer (Task 3 p. 26)
            nlg_info = generate_factor_nlg_summary(filtered_df, factor_col=selected_factor_col)
            st.markdown(f"**Automated Relationship Summary (`{selected_factor_label}`):**")
            for bullet in nlg_info["bullets"]:
                st.markdown(f"- {bullet}")

        # Maximized View — 4 Diagnostic Tabs (Task 3 p. 27-28)
        st.markdown("#### 🔍 Maximized Diagnostic View (SAS Visual Analytics Automated Explanation Details)")
        diag_t1, diag_t2, diag_t3, diag_t4 = st.tabs(
            [
                "1. Explanation Description",
                "2. Screening Results (3 of 17 Removed)",
                "3. Relative Importance Table",
                "4. High / Low Probability Groups & Anomalies",
            ]
        )
        with diag_t1:
            st.markdown(
                """
                **SAS Viya Automated Explanation Three-Stage Analytical Pipeline (Guide p. 27):**
                1. **Data Screening & Hygiene:** Evaluates all 17 input variables assigned to the explanation object. Variables with excessive cardinality (`City` with 858 levels), strict hierarchical redundancy (`Region`), or pure time-index behavior (`Loan Application Date`) are excluded before tree induction.
                2. **Single-Predictor & Surrogate Tree Modeling:** Fits regularized shallow decision trees (`max_depth=2..3`) with explicit missing-value branch assignment for each candidate factor to quantify information gain toward `Loan Status`.
                3. **Relative Importance Normalization & Segment Extraction:** Normalizes predictive gain scores so the strongest driver (**Debt to Income Ratio**) equals **1.00**, followed by **Number of Delinquent Credit Lines (0.35)** and **Value of Current Property (0.28)**, and extracts leaf rules with the highest and lowest event rates.
                """
            )
        with diag_t2:
            st.dataframe(explanation["screening_df"], use_container_width=True, hide_index=True)
        with diag_t3:
            st.dataframe(imp_df, use_container_width=True, hide_index=True)
        with diag_t4:
            hl_col1, hl_col2 = st.columns([1.25, 1.0])
            with hl_col1:
                st.markdown("**Decision Tree High / Low Probability Segments**")
                st.dataframe(explanation["high_low_df"], use_container_width=True, hide_index=True)
            with hl_col2:
                st.markdown("**Measure Anomalies & Extreme Outlier Summary**")
                meas_summary = compute_measure_summary_table(filtered_df)
                st.dataframe(
                    meas_summary[["SAS Label", "Average", "Minimum", "Maximum", "Missing", "Extreme Outliers"]],
                    use_container_width=True,
                    hide_index=True,
                )

    # =========================================================================
    # TAB 3: MODEL STUDIO, FAIRNESS & MODEL CARD (TASKS 4 & 5, GUIDE P. 28-45)
    # =========================================================================
    with tab3:
        model_results = train_and_evaluate_sas_models(full_df)
        comp_df = model_results["comparison_df"]
        part_df = model_results["partition_df"]
        job_fair_df = model_results["job_fairness_df"]

        # Extract Champion metrics for Model Card header
        champ_val = part_df[
            (part_df["Model"] == "Gradient Boosting (Champion)") & (part_df["Partition"] == "Validate (30%)")
        ].iloc[0]
        champ_test = part_df[
            (part_df["Model"] == "Gradient Boosting (Champion)") & (part_df["Partition"] == "Test (10%)")
        ].iloc[0]

        st.markdown("#### 🪪 Task 5: SAS Model Manager — Champion Model Card (`Gradient Boosting — Pipeline 1`)")
        render_model_card_health_row(
            champion_ks=float(champ_val["KS (Youden)"]),
            val_ase=float(champ_val["ASE"]),
            test_ase=float(champ_test["ASE"]),
            max_pred_diff_pct=model_results["max_pred_diff_pct"],
        )

        st.markdown("<div style='height:12px'></div>", unsafe_allow_html=True)

        st.markdown("#### 🏆 Task 4: Pipeline Comparison (Pipeline 1 AutoML vs. Pipeline 2 Open Source Python)")
        st.dataframe(comp_df, use_container_width=True, hide_index=True)

        mc_left, mc_right = st.columns(2)
        with mc_left:
            st.plotly_chart(create_roc_comparison_chart(model_results["roc_curves"]), use_container_width=True)
        with mc_right:
            st.plotly_chart(create_cumulative_lift_chart(model_results["lift_curves"]), use_container_width=True)

        st.markdown("---")
        st.markdown("#### ⚖️ Sensitive Variable Assessment: Fairness & Bias on `Job Category (JOB)` (Guide p. 34–36, 43–45)")
        fair_left, fair_right = st.columns([1.25, 1.0])
        with fair_left:
            st.plotly_chart(create_job_fairness_chart(job_fair_df), use_container_width=True)
        with fair_right:
            st.markdown(
                f"""
                **Why `Fairness` Evaluates to `FAIL` on the SAS Model Card (Guide p. 43–45):**
                - **Sensitive Attribute Monitored:** `JOB` (*Job Category*: `Mgr`, `Office`, `Other`, `ProfExe`, `Sales`, `Self`).
                - **Prediction Bias Spread:** **{model_results['max_pred_diff_pct']:.2f}%** difference in average predicted default probability between the highest-scored group (`Sales`) and lowest-scored group (`Office` / `ProfExe`).
                - **Equal Opportunity (TPR) Disparity:** **{model_results['max_tpr_diff_pct']:.2f}%** gap across job categories.
                - **Accuracy Spread Across Groups:** **{model_results['max_acc_diff_pct']:.2f}%** difference across job categories.
                - **Governance Recommendation:** Require secondary human/policy review (`Check Debt to Income` rule set in Task 6) and exclude raw occupational category from adverse-action standalone cutoffs.
                """
            )
            st.dataframe(
                job_fair_df[
                    [
                        "Job Category (JOB)",
                        "Applications",
                        "Actual Default Rate (%)",
                        "Average Predicted P(BAD=1) (%)",
                        "True Positive Rate (Equal Opportunity %)",
                        "Group KS",
                    ]
                ],
                use_container_width=True,
                hide_index=True,
            )

        with st.expander("📑 View Full Train (60%) / Validate (30%) / Test (10%) Partition Fit Statistics"):
            st.dataframe(part_df, use_container_width=True, hide_index=True)

    # =========================================================================
    # TAB 4: LOAN DECISION (FINAL) FLOW & SAS DATA PREP (TASKS 1 & 6, P. 14-16, 45-54)
    # =========================================================================
    with tab4:
        model_results = train_and_evaluate_sas_models(full_df)
        scored_df = model_results["scored_df"]
        champion_pipe = model_results["champion_pipeline"]

        st.markdown("#### 🔀 Task 6: SAS Intelligent Decisioning — `Loan Decision (Final)` Interactive Simulator")
        st.caption(
            "Replicates the exact decision flow diagram from Guide Figure 84 (p. 50): "
            "`Start` → `Gradient Boosting (Pipeline 1)` → Condition `P_BAD1 < 0.20` → "
            "Rule Set `Check Debt to Income (DEBTINC < 40)` / `Declined Applications` → "
            "Python Code Node `Create Email Message (Final)` → `End`."
        )

        sim_col1, sim_col2 = st.columns([1.15, 1.25])
        with sim_col1:
            st.markdown("##### 🧪 Configure Applicant & Decision Policy Thresholds")
            p_col1, p_col2 = st.columns(2)
            with p_col1:
                prob_cutoff = st.slider(
                    "ML Risk Cutoff (`P_BAD1 <`)",
                    min_value=0.05,
                    max_value=0.50,
                    value=0.20,
                    step=0.01,
                    help="SAS Guide Task 6 default: P_BAD1 < 0.20 passes to the Debt-to-Income check.",
                )
                sim_loan = st.number_input("Amount of Loan Request ($)", min_value=1000, max_value=100000, value=18500, step=500)
                sim_value = st.number_input("Value of Current Property ($)", min_value=10000, max_value=850000, value=115000, step=2500)
                sim_mortdue = st.number_input("Amount Due on Existing Mortgage ($)", min_value=0, max_value=600000, value=68000, step=2000)
                sim_debtinc = st.slider("Debt to Income Ratio (`DEBTINC` %)", min_value=5.0, max_value=85.0, value=33.5, step=0.5)
                sim_clage = st.slider("Age of Oldest Credit Line (months)", min_value=6.0, max_value=600.0, value=180.0, step=6.0)

            with p_col2:
                debtinc_cutoff = st.slider(
                    "DTI Policy Cap (`DEBTINC <`)",
                    min_value=25.0,
                    max_value=60.0,
                    value=40.0,
                    step=1.0,
                    help="SAS Guide Task 6 default: DEBTINC < 40 required for final loan approval.",
                )
                sim_delinq = st.number_input("Number of Delinquent Credit Lines (`DELINQ`)", min_value=0, max_value=15, value=0, step=1)
                sim_derog = st.number_input("Number of Derogatory Reports (`DEROG`)", min_value=0, max_value=15, value=0, step=1)
                sim_ninq = st.number_input("Number of Recent Credit Inquiries (`NINQ`)", min_value=0, max_value=15, value=1, step=1)
                sim_clno = st.number_input("Number of Credit Lines (`CLNO`)", min_value=0, max_value=65, value=21, step=1)
                sim_yoj = st.number_input("Years at Present Job (`YOJ`)", min_value=0.0, max_value=45.0, value=8.0, step=1.0)

            c_cat1, c_cat2, c_cat3 = st.columns(3)
            with c_cat1:
                sim_reason = st.selectbox("Loan Purpose", options=["DebtCon", "HomeImp"], index=0)
            with c_cat2:
                sim_job = st.selectbox("Job Category", options=["ProfExe", "Office", "Mgr", "Other", "Self", "Sales"], index=0)
            with c_cat3:
                sim_region = st.selectbox("Region", options=["South", "Midwest", "Northeast", "West"], index=0)

        applicant_payload = {
            "LOAN": sim_loan,
            "MORTDUE": sim_mortdue,
            "VALUE": sim_value,
            "YOJ": sim_yoj,
            "DEROG": sim_derog,
            "DELINQ": sim_delinq,
            "CLAGE": sim_clage,
            "NINQ": sim_ninq,
            "CLNO": sim_clno,
            "DEBTINC": sim_debtinc,
            "REASON": sim_reason,
            "JOB": sim_job,
            "REGION": sim_region,
            "DIVISION": "South Atlantic",
        }
        decision_out = evaluate_decision_flow_single(
            champion_pipe,
            applicant_payload,
            prob_cutoff=prob_cutoff,
            debtinc_cutoff=debtinc_cutoff,
        )

        with sim_col2:
            st.markdown("##### 📬 Real-Time Decision Execution & `Create Email Message (Final)` Output")
            status_badge_color = "#1E8E3E" if decision_out["status"] == "Approved" else "#D9383A"
            status_bg_color = "#E6F4EA" if decision_out["status"] == "Approved" else "#FCE8E6"
            st.markdown(
                f"""
                <div style="background:{status_bg_color};border:1px solid {status_badge_color};border-left:6px solid {status_badge_color};padding:14px 18px;border-radius:8px;margin-bottom:12px;">
                    <div style="font-size:1.15rem;font-weight:700;color:{status_badge_color};">
                        Decision Outcome: {decision_out['status'].upper()}
                    </div>
                    <div style="font-size:0.88rem;color:#1D242B;margin-top:4px;">
                        <b>Executed Node:</b> <code>{decision_out['active_node']}</code><br>
                        <b>Model Score (P_BAD1):</b> <code>{decision_out['p_bad1']:.2%}</code> (Branch Condition: <code>P_BAD1 &lt; {prob_cutoff:.0%}</code> → <b>{'YES' if decision_out['passed_ml_branch'] else 'NO'}</b>)<br>
                        <b>Debt-to-Income Check:</b> <code>{sim_debtinc:.1f}%</code> (Rule Condition: <code>DEBTINC &lt; {debtinc_cutoff:.0f}%</code> → <b>{'PASS' if decision_out['passed_dti_rule'] else 'FAIL'}</b>)<br>
                        <b>Calculated LTV:</b> <code>{(sim_loan / sim_value * 100.0):.1f}%</code>
                    </div>
                </div>
                """,
                unsafe_allow_html=True,
            )
            st.markdown(f"**Subject:** `{decision_out['email_subject']}`")
            st.code(decision_out["email_body"], language="text")

        st.markdown("---")
        st.markdown("#### 📈 Portfolio-Wide Batch Decisioning Simulation (`5,960` Applications)")
        batch_res = evaluate_batch_decision_flow(scored_df, prob_cutoff=prob_cutoff, debtinc_cutoff=debtinc_cutoff)

        b_kpi1, b_kpi2, b_kpi3, b_kpi4 = st.columns(4)
        b_kpi1.metric("Approved Applications", f"{batch_res['approved_count']:,}", f"{batch_res['approved_pct']:.1f}% approval rate")
        b_kpi2.metric("Observed Default Rate in Approved Pool", f"{batch_res['approved_default_rate']:.2f}%", "vs 19.95% baseline", delta_color="off")
        b_kpi3.metric("Declined at ML Branch (P_BAD1 >= Cutoff)", f"{batch_res['declined_ml_count']:,}", f"{batch_res['declined_ml_pct']:.1f}% of total", delta_color="off")
        b_kpi4.metric("Declined at DTI Rule Set (DEBTINC >= Cap)", f"{batch_res['declined_dti_count']:,}", f"{batch_res['declined_dti_pct']:.1f}% of total", delta_color="off")

        sankey_col, table_col = st.columns([1.15, 1.0])
        with sankey_col:
            st.plotly_chart(
                create_decision_flow_sankey(batch_res, prob_cutoff=prob_cutoff, debtinc_cutoff=debtinc_cutoff),
                use_container_width=True,
            )
        with table_col:
            st.markdown("**Batch Decision Stage Breakdown**")
            st.dataframe(batch_res["summary_table"], use_container_width=True, hide_index=True)

        st.markdown("---")
        with st.expander("🛠️ Task 1: SAS Studio Data Preparation (`Import_and_load_HOMEEQUITY1.sas`) & Variable Dictionary", expanded=False):
            st.markdown(
                """
                **SAS Studio Data Preparation Steps Replicated in `src/data_engine.py` (Guide p. 14–16):**
                - Assigned all 18 human-readable SAS Variable Labels (`BAD` → *Loan Status*, `LOAN` → *Amount of Loan Request*, etc.).
                - Converted `CITY` to Proper Case (`CITY = propcase(CITY)`).
                - Created Calculated Item `LTV = LOAN / VALUE` (*Loan to Value ratio*).
                - Formatted SAS numeric `APPDATE` (days since `1960-01-01`, range `20820` to `22645`) into `MMMYYYY (MONYY7)` (`JAN2017` – `DEC2021`).
                - Converted two-level numeric `BAD` (`0` / `1`) to discrete category `Loan Status` and set default aggregation of all 11 continuous measures to `Average`.
                """
            )
            st.dataframe(compute_measure_summary_table(filtered_df), use_container_width=True, hide_index=True)


if __name__ == "__main__":
    main()
