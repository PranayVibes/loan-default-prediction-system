from __future__ import annotations

import json
from pathlib import Path

import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import streamlit as st

from src.predict import predict_default_risk
from src.train_model import FEATURE_COLUMNS, train_and_save


st.set_page_config(
    page_title="Loan Default Prediction",
    page_icon="🏦",
    layout="wide",
)

ARTIFACTS_DIR = Path("artifacts")
METRICS_PATH  = ARTIFACTS_DIR / "model_metrics.json"

# Dark-theme friendly palette
APPROVE_COLOR = "#00c896"   # teal-green
REVIEW_COLOR  = "#f5a623"   # amber
REJECT_COLOR  = "#ff4b4b"   # soft red
BLUE_COLOR    = "#4c9be8"   # info blue
BG_COLOR      = "#0e1117"   # streamlit dark bg
CARD_COLOR    = "#1e2130"   # slightly lighter card bg
TEXT_COLOR    = "#fafafa"

# Matplotlib dark style for all charts
plt.rcParams.update({
    "figure.facecolor":  "#1e2130",
    "axes.facecolor":    "#1e2130",
    "axes.edgecolor":    "#3a3f55",
    "axes.labelcolor":   "#e0e0e0",
    "xtick.color":       "#b0b0b0",
    "ytick.color":       "#b0b0b0",
    "text.color":        "#e0e0e0",
    "grid.color":        "#2e3348",
    "grid.linewidth":    0.8,
    "axes.titlecolor":   "#ffffff",
    "axes.titlesize":    13,
    "axes.titleweight":  "bold",
    "axes.labelsize":    11,
    "font.family":       "DejaVu Sans",
    "legend.facecolor":  "#1e2130",
    "legend.edgecolor":  "#3a3f55",
    "legend.labelcolor": "#e0e0e0",
})

# ── Global CSS ─────────────────────────────────────────────────────────────────
st.markdown("""
<style>
/* ── Main title ── */
.main-title {
    text-align: center;
    font-size: 2.6rem;
    font-weight: 800;
    color: #ffffff;
    letter-spacing: 0.5px;
    margin-bottom: 0.1rem;
}
.main-subtitle {
    text-align: center;
    font-size: 1.05rem;
    color: #9ba3b8;
    margin-bottom: 1.2rem;
}

/* ── Section headings ── */
.section-heading {
    font-size: 1.45rem;
    font-weight: 700;
    color: #ffffff;
    border-left: 4px solid #4c9be8;
    padding-left: 10px;
    margin-top: 1.4rem;
    margin-bottom: 0.6rem;
}
.sub-heading {
    font-size: 1.15rem;
    font-weight: 600;
    color: #d0d6e8;
    margin-top: 0.8rem;
    margin-bottom: 0.3rem;
}

/* ── Chart title inside columns ── */
.chart-title {
    font-size: 1.05rem;
    font-weight: 700;
    color: #e8eaf6;
    text-align: center;
    margin-bottom: 0.2rem;
}
.chart-caption {
    font-size: 0.82rem;
    color: #8a93aa;
    text-align: center;
    margin-top: 0.2rem;
}

/* ── Divider ── */
.styled-divider {
    border: none;
    border-top: 1px solid #2e3348;
    margin: 1.4rem 0;
}

/* ── Metric labels larger ── */
[data-testid="stMetricLabel"] {
    font-size: 0.9rem !important;
    font-weight: 600 !important;
    color: #9ba3b8 !important;
}
[data-testid="stMetricValue"] {
    font-size: 1.5rem !important;
    font-weight: 800 !important;
}

/* ── Tab font ── */
[data-testid="stTabs"] button {
    font-size: 1.0rem !important;
    font-weight: 600 !important;
}
</style>
""", unsafe_allow_html=True)


# ── Helpers ────────────────────────────────────────────────────────────────────
def ensure_artifacts() -> dict:
    if not METRICS_PATH.exists():
        return train_and_save()
    return json.loads(METRICS_PATH.read_text())


def risk_badge(p: float) -> str:
    return "Low Risk" if p < 0.3 else ("Medium Risk" if p < 0.6 else "High Risk")

def decision_badge(p: float) -> str:
    return "Approve" if p < 0.3 else ("Review" if p < 0.6 else "Reject")


# ── Bootstrap ──────────────────────────────────────────────────────────────────
metrics = ensure_artifacts()

st.markdown('<div class="main-title">🏦 Loan Default Prediction System</div>', unsafe_allow_html=True)
st.markdown('<div class="main-subtitle">DSPYL Lab Mini Project &nbsp;·&nbsp; Bank / NBFC Loan Approval Support</div>', unsafe_allow_html=True)

c1, c2, c3, c4 = st.columns(4)
c1.metric("🏆 Best Model",    metrics.get("best_model", "—"))
c2.metric("🎯 Accuracy",      f"{metrics['accuracy'] * 100:.2f}%")
c3.metric("📈 ROC-AUC",       f"{metrics['roc_auc']:.4f}")
c4.metric("🎯 Business Goal", "Approve low-risk only")

st.markdown('<hr class="styled-divider">', unsafe_allow_html=True)

tab1, tab2, tab3 = st.tabs(["  🔍 Single Prediction  ", "  📊 Batch Prediction  ", "  📋 Project Overview  "])


# ══════════════════════════════════════════════════════════════════════════════
# TAB 1 — Single Prediction
# ══════════════════════════════════════════════════════════════════════════════
with tab1:
    st.markdown('<div class="section-heading">Customer Risk Checker</div>', unsafe_allow_html=True)

    with st.form("loan_form"):
        c1, c2, c3 = st.columns(3)
        age              = c1.slider("Age", 21, 60, 30)
        annual_income    = c2.number_input("Annual Income (₹)", min_value=180000, max_value=1800000, value=600000, step=10000)
        loan_amount      = c3.number_input("Loan Amount (₹)",   min_value=50000,  max_value=1000000, value=250000, step=10000)

        c4, c5, c6 = st.columns(3)
        credit_score     = c4.slider("Credit Score",       300, 850, 700)
        employment_years = c5.slider("Employment Years",   0,   20,  5)
        existing_loans   = c6.slider("Existing Loans",     0,   5,   1)

        c7, c8, c9 = st.columns(3)
        dependents           = c7.slider("Dependents",               0,   4,   1)
        missed_payments      = c8.slider("Missed Payments",          0,   6,   0)
        debt_to_income_ratio = c9.slider("Debt-to-Income Ratio (%)", 3.0, 95.0, 30.0, 0.5)

        c10, c11, c12 = st.columns(3)
        education       = c10.selectbox("Education",       ["Graduate", "Postgraduate", "High School", "Diploma"])
        employment_type = c11.selectbox("Employment Type", ["Salaried", "Self-Employed", "Business", "Unemployed"])
        residence_type  = c12.selectbox("Residence Type",  ["Owned", "Rented", "Mortgaged"])

        c13, c14 = st.columns(2)
        marital_status = c13.selectbox("Marital Status", ["Single", "Married", "Divorced"])
        loan_purpose   = c14.selectbox("Loan Purpose",   ["Home", "Vehicle", "Education", "Business", "Personal"])

        submitted = st.form_submit_button("🔮 Predict Loan Risk", use_container_width=True)

    if submitted:
        input_df = pd.DataFrame([{
            "age": age, "annual_income": annual_income, "loan_amount": loan_amount,
            "credit_score": credit_score, "employment_years": employment_years,
            "existing_loans": existing_loans, "dependents": dependents,
            "missed_payments": missed_payments, "debt_to_income_ratio": debt_to_income_ratio,
            "education": education, "employment_type": employment_type,
            "residence_type": residence_type, "marital_status": marital_status,
            "loan_purpose": loan_purpose,
        }])
        result      = predict_default_risk(input_df).iloc[0]
        probability = float(result["default_probability"])

        st.markdown('<hr class="styled-divider">', unsafe_allow_html=True)
        st.markdown('<div class="section-heading">Prediction Result</div>', unsafe_allow_html=True)

        a1, a2, a3 = st.columns(3)
        a1.metric("Default Probability", f"{probability * 100:.2f}%")
        a2.metric("Risk Level",          risk_badge(probability))
        a3.metric("Decision",            decision_badge(probability))

        if probability < 0.3:
            st.success(f"✅ **Low Risk** — Default probability is **{probability*100:.1f}%**. This customer can be approved for the loan.")
        elif probability < 0.6:
            st.warning(f"⚠️ **Medium Risk** — Default probability is **{probability*100:.1f}%**. This application should be reviewed manually before a final decision.")
        else:
            st.error(f"❌ **High Risk** — Default probability is **{probability*100:.1f}%**. This customer poses significant default risk and should be rejected.")

        st.markdown("""
        <div style='background:#1e2130;border-radius:8px;padding:10px 16px;margin-top:10px;font-size:0.88rem;color:#9ba3b8;'>
        <b style='color:#d0d6e8;'>Decision Thresholds:</b>
        &nbsp; 🟢 &lt; 30% → <b>Approve</b>
        &nbsp;|&nbsp; 🟡 30–60% → <b>Review</b>
        &nbsp;|&nbsp; 🔴 ≥ 60% → <b>Reject</b>
        </div>
        """, unsafe_allow_html=True)


# ══════════════════════════════════════════════════════════════════════════════
# TAB 2 — Batch Prediction
# ══════════════════════════════════════════════════════════════════════════════
with tab2:
    st.markdown('<div class="section-heading">Batch Prediction from CSV</div>', unsafe_allow_html=True)
    st.markdown(f"**Required columns:** `{', '.join(FEATURE_COLUMNS)}`")
    uploaded_file = st.file_uploader("Upload customer CSV file", type=["csv"])

    if uploaded_file is not None:
        batch_df = pd.read_csv(uploaded_file)
        missing_columns = [c for c in FEATURE_COLUMNS if c not in batch_df.columns]

        if missing_columns:
            st.error(f"Missing columns: {', '.join(missing_columns)}")
        else:
            batch_result = predict_default_risk(batch_df)
            full_df = pd.concat(
                [batch_df.reset_index(drop=True),
                 batch_result[["default_probability", "predicted_default", "risk_level", "decision"]].reset_index(drop=True)],
                axis=1,
            )

            # ── 1. Prediction Table ───────────────────────────────────────────
            st.markdown('<hr class="styled-divider">', unsafe_allow_html=True)
            st.markdown('<div class="section-heading">1. Prediction Results</div>', unsafe_allow_html=True)
            st.dataframe(batch_result, use_container_width=True)
            st.download_button(
                "⬇️ Download Predictions CSV",
                batch_result.to_csv(index=False).encode("utf-8"),
                file_name="loan_predictions.csv",
                mime="text/csv",
            )

            # ── 2. Summary KPIs ───────────────────────────────────────────────
            st.markdown('<hr class="styled-divider">', unsafe_allow_html=True)
            st.markdown('<div class="section-heading">2. Summary</div>', unsafe_allow_html=True)

            total      = len(full_df)
            approved   = int((full_df["risk_level"] == "Low Risk").sum())
            review     = int((full_df["risk_level"] == "Medium Risk").sum())
            rejected   = int((full_df["risk_level"] == "High Risk").sum())
            avg_prob   = float(full_df["default_probability"].mean())
            avg_credit = int(full_df["credit_score"].mean())

            m1, m2, m3, m4, m5, m6 = st.columns(6)
            m1.metric("Total Applicants", total)
            m2.metric("✅ Approved",      approved)
            m3.metric("⚠️ For Review",   review)
            m4.metric("❌ Rejected",      rejected)
            m5.metric("Avg Default Prob", f"{avg_prob * 100:.1f}%")
            m6.metric("Avg Credit Score", avg_credit)

            # ── 3. Key Insights ───────────────────────────────────────────────
            st.markdown('<hr class="styled-divider">', unsafe_allow_html=True)
            st.markdown('<div class="section-heading">3. Key Insights</div>', unsafe_allow_html=True)

            high_risk_pct         = rejected / total * 100
            low_income_cnt        = int((full_df["annual_income"] < 400000).sum())
            low_credit_cnt        = int((full_df["credit_score"]  < 600).sum())
            avg_income_default    = full_df[full_df["predicted_default"] == 1]["annual_income"].mean()
            avg_income_no_default = full_df[full_df["predicted_default"] == 0]["annual_income"].mean()
            avg_cs_default        = full_df[full_df["predicted_default"] == 1]["credit_score"].mean()
            avg_cs_no_default     = full_df[full_df["predicted_default"] == 0]["credit_score"].mean()

            st.markdown(f"""
1. **High-Risk Applicants:** **{rejected}** out of **{total}** ({high_risk_pct:.1f}%) are classified as high risk (default probability ≥ 60%). These should be rejected to minimise credit loss.

2. **Credit Score & Default:** Defaulted applicants averaged a credit score of **{avg_cs_default:.0f}**, versus **{avg_cs_no_default:.0f}** for non-defaulters. A score below 600 is the strongest single predictor of default.

3. **Missed Payments Impact:** **{low_credit_cnt}** applicants have a credit score below 600. Each additional missed payment substantially increases the probability of default — applicants with 3+ missed payments are high-risk.

4. **Income vs Default:** Defaulted applicants had a mean annual income of **₹{avg_income_default:,.0f}**, compared to **₹{avg_income_no_default:,.0f}** for non-defaulters. **{low_income_cnt}** applicants earn below ₹4,00,000 — lower income means higher financial stress and elevated default risk.
            """)

            # ── 4. Model Comparison ───────────────────────────────────────────
            st.markdown('<hr class="styled-divider">', unsafe_allow_html=True)
            st.markdown('<div class="section-heading">4. Model Comparison</div>', unsafe_allow_html=True)

            comparison = metrics.get("comparison", {})
            if comparison:
                comp_df = pd.DataFrame(comparison).T.reset_index()
                comp_df.columns = ["Model", "Accuracy", "ROC-AUC"]
                comp_df["Accuracy"] = comp_df["Accuracy"].apply(lambda v: f"{v * 100:.2f}%")
                comp_df["ROC-AUC"]  = comp_df["ROC-AUC"].apply(lambda v: f"{v:.4f}")
                best_name = metrics.get("best_model", "")
                comp_df["Selected"] = comp_df["Model"].apply(lambda m: "✅ Best Model" if m == best_name else "")
                st.dataframe(comp_df, use_container_width=True, hide_index=True)
                st.caption(f"The **{best_name}** was selected as the production model based on higher ROC-AUC score.")

            # ── 5. Confusion Matrix ───────────────────────────────────────────
            st.markdown('<hr class="styled-divider">', unsafe_allow_html=True)
            st.markdown('<div class="section-heading">5. Confusion Matrix</div>', unsafe_allow_html=True)

            cm_data = metrics.get("confusion_matrix")
            if cm_data:
                cm_arr = np.array(cm_data)
                tn, fp, fn, tp = cm_arr.ravel()

                cm_col1, cm_col2 = st.columns([1, 1])
                with cm_col1:
                    fig_cm, ax_cm = plt.subplots(figsize=(5, 4))
                    annot_labels = np.array([[f"TN\n{tn}", f"FP\n{fp}"],
                                             [f"FN\n{fn}", f"TP\n{tp}"]])
                    sns.heatmap(
                        cm_arr,
                        annot=annot_labels,
                        fmt="",
                        cmap="Blues",
                        ax=ax_cm,
                        linewidths=2,
                        linecolor="#0e1117",
                        cbar=False,
                        annot_kws={"size": 15, "weight": "bold", "color": "white"},
                    )
                    ax_cm.set_xlabel("Predicted Label", fontsize=11, labelpad=8)
                    ax_cm.set_ylabel("Actual Label", fontsize=11, labelpad=8)
                    ax_cm.set_xticklabels(["No Default", "Default"], fontsize=10)
                    ax_cm.set_yticklabels(["No Default", "Default"], fontsize=10, rotation=0)
                    ax_cm.set_title(f"Confusion Matrix — {metrics.get('best_model','')}", fontsize=13, fontweight="bold", pad=12, color="white")
                    fig_cm.tight_layout()
                    st.pyplot(fig_cm, use_container_width=True)
                    plt.close(fig_cm)

                with cm_col2:
                    st.markdown('<div class="sub-heading">What these numbers mean</div>', unsafe_allow_html=True)
                    st.markdown(f"""
| Metric | Value | Meaning |
|---|---|---|
| **True Positive (TP)** | **{tp}** | Correctly predicted defaulters |
| **True Negative (TN)** | **{tn}** | Correctly predicted non-defaulters |
| **False Positive (FP)** | **{fp}** | Non-defaulters wrongly flagged as risky |
| **False Negative (FN)** | **{fn}** | Defaulters the model missed |

- A high **FN** means the bank approved risky customers — costly for the bank.
- A high **FP** means good customers were incorrectly rejected — bad for business.
- Minimising **FN** is the priority in credit risk models.
                    """)

            # ── 6. ROC Curve ──────────────────────────────────────────────────
            st.markdown('<hr class="styled-divider">', unsafe_allow_html=True)
            st.markdown('<div class="section-heading">6. ROC Curve</div>', unsafe_allow_html=True)

            roc_fpr     = metrics.get("roc_fpr")
            roc_tpr     = metrics.get("roc_tpr")
            roc_auc_val = metrics.get("roc_auc", 0)

            if roc_fpr and roc_tpr:
                roc_col, _ = st.columns([2, 1])
                with roc_col:
                    fig_roc, ax_roc = plt.subplots(figsize=(7, 5))
                    ax_roc.plot(roc_fpr, roc_tpr, color="#4c9be8", lw=2.5,
                                label=f"ROC Curve  (AUC = {roc_auc_val:.4f})")
                    ax_roc.plot([0, 1], [0, 1], color="#555e7a", lw=1.8,
                                linestyle="--", label="Random Classifier  (AUC = 0.50)")
                    ax_roc.fill_between(roc_fpr, roc_tpr, alpha=0.12, color="#4c9be8")
                    ax_roc.set_xlabel("False Positive Rate", fontsize=12, labelpad=8)
                    ax_roc.set_ylabel("True Positive Rate", fontsize=12, labelpad=8)
                    ax_roc.set_title(f"ROC Curve — {metrics.get('best_model','Best Model')}",
                                     fontsize=14, fontweight="bold", pad=12)
                    ax_roc.legend(fontsize=11, loc="lower right")
                    ax_roc.set_xlim([0.0, 1.0])
                    ax_roc.set_ylim([0.0, 1.02])
                    # AUC annotation box
                    ax_roc.annotate(
                        f"AUC = {roc_auc_val:.4f}",
                        xy=(0.6, 0.25), fontsize=13, fontweight="bold",
                        color="#4c9be8",
                        bbox=dict(boxstyle="round,pad=0.4", fc="#1e2130", ec="#4c9be8", lw=1.5),
                    )
                    fig_roc.tight_layout()
                    st.pyplot(fig_roc, use_container_width=True)
                    plt.close(fig_roc)
                st.caption(
                    f"AUC = {roc_auc_val:.4f} — The model correctly ranks a random defaulter above a non-defaulter "
                    f"{roc_auc_val*100:.1f}% of the time. Values above 0.80 are strong for credit risk."
                )

            # ── 7. Visual Analysis ────────────────────────────────────────────
            st.markdown('<hr class="styled-divider">', unsafe_allow_html=True)
            st.markdown('<div class="section-heading">7. Visual Analysis</div>', unsafe_allow_html=True)

            def bar_label(ax, bars, total_n, fmt="{v}  ({pct:.1f}%)", fontsize=11):
                """Add value + percentage labels on top of each bar."""
                for bar in bars:
                    v = bar.get_height()
                    pct = v / total_n * 100
                    ax.text(
                        bar.get_x() + bar.get_width() / 2,
                        v + ax.get_ylim()[1] * 0.01,
                        fmt.format(v=int(v), pct=pct),
                        ha="center", va="bottom",
                        fontsize=fontsize, fontweight="bold", color="#ffffff",
                    )

            chart_col1, chart_col2 = st.columns(2)

            # Chart A: Default vs Non-Default
            with chart_col1:
                st.markdown('<div class="chart-title">Default vs Non-Default Distribution</div>', unsafe_allow_html=True)
                counts = full_df["predicted_default"].value_counts().sort_index()
                fig1, ax1 = plt.subplots(figsize=(5, 4.2))
                bars1 = ax1.bar(
                    ["Non-Default\n(Approved)", "Default\n(Rejected)"],
                    counts.values,
                    color=[APPROVE_COLOR, REJECT_COLOR],
                    width=0.48, edgecolor="#0e1117", linewidth=1.5,
                )
                bar_label(ax1, bars1, total)
                ax1.set_title("Default vs Non-Default", fontsize=13, fontweight="bold", pad=10)
                ax1.set_ylabel("Number of Applicants", fontsize=11)
                ax1.set_ylim(0, max(counts.values) * 1.28)
                ax1.tick_params(axis="x", labelsize=10)
                fig1.tight_layout()
                st.pyplot(fig1, use_container_width=True)
                plt.close(fig1)
                st.markdown('<div class="chart-caption">Overall split between predicted defaulters and non-defaulters — a quick portfolio health check.</div>', unsafe_allow_html=True)

            # Chart B: Income Distribution
            with chart_col2:
                st.markdown('<div class="chart-title">Income Distribution by Default Status</div>', unsafe_allow_html=True)
                fig2, ax2 = plt.subplots(figsize=(5, 4.2))
                for val, label, color in zip([0, 1], ["Non-Default", "Default"], [APPROVE_COLOR, REJECT_COLOR]):
                    subset = full_df[full_df["predicted_default"] == val]["annual_income"]
                    ax2.hist(subset, bins=20, alpha=0.72, color=color, label=label, edgecolor="#0e1117", linewidth=0.6)
                ax2.set_title("Income Distribution by Default Status", fontsize=13, fontweight="bold", pad=10)
                ax2.set_xlabel("Annual Income (₹)", fontsize=11)
                ax2.set_ylabel("Number of Applicants", fontsize=11)
                ax2.legend(fontsize=10)
                fig2.tight_layout()
                st.pyplot(fig2, use_container_width=True)
                plt.close(fig2)
                st.markdown('<div class="chart-caption">Lower-income applicants cluster more heavily in the default group — income is a strong repayment signal.</div>', unsafe_allow_html=True)

            chart_col3, chart_col4 = st.columns(2)

            # Chart C: Credit Score Box Plot
            with chart_col3:
                st.markdown('<div class="chart-title">Credit Score vs Default Status</div>', unsafe_allow_html=True)
                fig3, ax3 = plt.subplots(figsize=(5, 4.2))
                data_groups = [
                    full_df[full_df["predicted_default"] == 0]["credit_score"].values,
                    full_df[full_df["predicted_default"] == 1]["credit_score"].values,
                ]
                bp = ax3.boxplot(
                    data_groups,
                    labels=["Non-Default", "Default"],
                    patch_artist=True,
                    medianprops=dict(color="white", linewidth=2.5),
                    boxprops=dict(linewidth=1.5),
                    whiskerprops=dict(linewidth=1.2, color="#8a93aa"),
                    capprops=dict(linewidth=1.2, color="#8a93aa"),
                    flierprops=dict(marker="o", markersize=5, markerfacecolor="#8a93aa", alpha=0.5),
                )
                bp["boxes"][0].set_facecolor(APPROVE_COLOR)
                bp["boxes"][1].set_facecolor(REJECT_COLOR)
                ax3.set_title("Credit Score vs Default Status", fontsize=13, fontweight="bold", pad=10)
                ax3.set_ylabel("Credit Score", fontsize=11)
                ax3.tick_params(axis="x", labelsize=10)
                for i, group in enumerate(data_groups):
                    med = pd.Series(group).median()
                    ax3.text(
                        i + 1, med + 5, f"Median: {med:.0f}",
                        ha="center", fontsize=9, fontweight="bold", color="white",
                        bbox=dict(boxstyle="round,pad=0.3", fc="#0e1117", ec="#4c9be8", lw=1.2, alpha=0.85),
                    )
                fig3.tight_layout()
                st.pyplot(fig3, use_container_width=True)
                plt.close(fig3)
                st.markdown('<div class="chart-caption">Defaulters consistently show lower credit scores. Below 600 is a strong default indicator.</div>', unsafe_allow_html=True)

            # Chart D: Missed Payments vs Default Rate
            with chart_col4:
                st.markdown('<div class="chart-title">Missed Payments vs Default Rate</div>', unsafe_allow_html=True)
                missed_dr = (
                    full_df.groupby("missed_payments")["predicted_default"]
                    .mean().mul(100).reset_index()
                )
                missed_dr.columns = ["Missed", "Rate"]

                fig4, ax4 = plt.subplots(figsize=(5, 4.2))
                bar_colors4 = [REJECT_COLOR if v >= 50 else APPROVE_COLOR for v in missed_dr["Rate"]]
                bars4 = ax4.bar(
                    missed_dr["Missed"].astype(str), missed_dr["Rate"],
                    color=bar_colors4, edgecolor="#0e1117", linewidth=1.2, width=0.55,
                )
                for bar, val in zip(bars4, missed_dr["Rate"]):
                    ax4.text(
                        bar.get_x() + bar.get_width() / 2,
                        bar.get_height() + 1.5,
                        f"{val:.1f}%",
                        ha="center", va="bottom",
                        fontsize=10, fontweight="bold", color="#ffffff",
                    )
                ax4.set_title("Missed Payments vs Default Rate", fontsize=13, fontweight="bold", pad=10)
                ax4.set_xlabel("Number of Missed Payments", fontsize=11)
                ax4.set_ylabel("Default Rate (%)", fontsize=11)
                ax4.set_ylim(0, 115)
                ax4.tick_params(axis="x", labelsize=10)
                approve_patch = mpatches.Patch(color=APPROVE_COLOR, label="< 50% default rate")
                reject_patch  = mpatches.Patch(color=REJECT_COLOR,  label="≥ 50% default rate")
                ax4.legend(handles=[approve_patch, reject_patch], fontsize=9)
                fig4.tight_layout()
                st.pyplot(fig4, use_container_width=True)
                plt.close(fig4)
                st.markdown('<div class="chart-caption">Each additional missed payment sharply raises default rate. 3+ missed payments = high risk.</div>', unsafe_allow_html=True)

            # Chart E: Default Rate by Loan Purpose (full width)
            st.markdown('<hr class="styled-divider">', unsafe_allow_html=True)
            st.markdown('<div class="chart-title" style="text-align:left;font-size:1.05rem;">Default Rate (%) by Loan Purpose</div>', unsafe_allow_html=True)

            purpose_dr = (
                full_df.groupby("loan_purpose")["predicted_default"]
                .mean().mul(100).round(1).sort_values(ascending=False)
            )
            fig5, ax5 = plt.subplots(figsize=(9, 4))
            bar_colors5 = [REJECT_COLOR if v > 50 else APPROVE_COLOR for v in purpose_dr.values]
            bars5 = ax5.bar(
                purpose_dr.index, purpose_dr.values,
                color=bar_colors5, edgecolor="#0e1117", linewidth=1.2, width=0.5,
            )
            for bar, val in zip(bars5, purpose_dr.values):
                ax5.text(
                    bar.get_x() + bar.get_width() / 2,
                    bar.get_height() + 1.2,
                    f"{val:.1f}%",
                    ha="center", va="bottom",
                    fontsize=11, fontweight="bold", color="#ffffff",
                )
            ax5.set_title("Default Rate (%) by Loan Purpose", fontsize=14, fontweight="bold", pad=12)
            ax5.set_ylabel("Default Rate (%)", fontsize=12)
            ax5.set_xlabel("Loan Purpose", fontsize=12)
            ax5.set_ylim(0, 105)
            ax5.tick_params(axis="x", labelsize=11)
            approve_p = mpatches.Patch(color=APPROVE_COLOR, label="≤ 50% (safer)")
            reject_p  = mpatches.Patch(color=REJECT_COLOR,  label="> 50% (risky)")
            ax5.legend(handles=[approve_p, reject_p], fontsize=10)
            fig5.tight_layout()
            st.pyplot(fig5, use_container_width=True)
            plt.close(fig5)


# ══════════════════════════════════════════════════════════════════════════════
# TAB 3 — Project Overview
# ══════════════════════════════════════════════════════════════════════════════
with tab3:
    st.markdown('<div class="section-heading">About This Project</div>', unsafe_allow_html=True)
    st.markdown("""
This system helps banks and NBFCs assess loan applications using machine learning.
Given applicant details — income, credit history, employment status, and repayment behavior —
the model estimates the probability of default and recommends a loan decision.
    """)

    st.markdown('<hr class="styled-divider">', unsafe_allow_html=True)
    st.markdown('<div class="sub-heading">Model Workflow</div>', unsafe_allow_html=True)
    st.markdown("""
```
Data  →  Preprocessing  →  Model Training  →  Prediction  →  Risk Decision
```
1. **Data** — Synthetic dataset of loan applicants with 14 features and a binary default label.
2. **Preprocessing** — Missing value imputation, feature scaling (StandardScaler), and one-hot encoding for categorical variables.
3. **Model Training** — Logistic Regression and Random Forest are both trained; the model with higher ROC-AUC is saved to production.
4. **Prediction** — The saved model outputs a default probability for each applicant.
5. **Decision** — A threshold-based rule converts the probability into a risk category and approval recommendation.
    """)

    st.markdown('<hr class="styled-divider">', unsafe_allow_html=True)
    st.markdown('<div class="sub-heading">Performance Metrics</div>', unsafe_allow_html=True)

    comparison = metrics.get("comparison", {})
    if comparison:
        comp_df = pd.DataFrame(comparison).T.reset_index()
        comp_df.columns = ["Model", "Accuracy", "ROC-AUC"]
        comp_df["Accuracy"] = comp_df["Accuracy"].apply(lambda v: f"{v * 100:.2f}%")
        comp_df["ROC-AUC"]  = comp_df["ROC-AUC"].apply(lambda v: f"{v:.4f}")
        best_name = metrics.get("best_model", "")
        comp_df["Selected"] = comp_df["Model"].apply(lambda m: "✅" if m == best_name else "")
        st.dataframe(comp_df, use_container_width=True, hide_index=True)

    st.markdown("""
- **Accuracy** — The percentage of applicants correctly classified. Easy to interpret but can be misleading on imbalanced datasets.
- **ROC-AUC** — Measures the model's ability to distinguish between defaulters and non-defaulters across all thresholds. A score of 1.0 is perfect; 0.5 is random. Above 0.80 is strong for credit risk.
    """)

    st.markdown('<hr class="styled-divider">', unsafe_allow_html=True)
    st.markdown('<div class="sub-heading">Decision Logic</div>', unsafe_allow_html=True)
    st.markdown("""
| Probability | Risk Level | Decision |
|---|---|---|
| < 30% | 🟢 Low Risk | **Approve** |
| 30% – 60% | 🟡 Medium Risk | **Review** |
| ≥ 60% | 🔴 High Risk | **Reject** |

The three-tier system gives relationship managers flexibility to manually review borderline cases.
    """)

    st.markdown('<hr class="styled-divider">', unsafe_allow_html=True)
    st.markdown('<div class="sub-heading">Input Features</div>', unsafe_allow_html=True)
    st.markdown("""
| Numeric Features | Categorical Features |
|---|---|
| Age, Annual Income, Loan Amount | Education, Employment Type |
| Credit Score, Employment Years | Residence Type, Marital Status |
| Existing Loans, Dependents | Loan Purpose |
| Missed Payments, Debt-to-Income Ratio | — |
    """)

    st.markdown('<hr class="styled-divider">', unsafe_allow_html=True)
    st.markdown('<div class="sub-heading">Limitations</div>', unsafe_allow_html=True)
    st.markdown("""
1. Trained on synthetic data — may not reflect real-world credit distributions perfectly.
2. Prediction quality depends on the accuracy and completeness of input data; missing or incorrect values can skew results.
3. The fixed thresholds (30% / 60%) may need adjustment based on the bank's risk appetite and portfolio composition.
4. Macroeconomic factors such as interest rate changes or economic downturns are not accounted for.
    """)

    st.markdown('<hr class="styled-divider">', unsafe_allow_html=True)
    st.caption("Built with Python · Scikit-learn · Pandas · Streamlit · Matplotlib · Seaborn")
