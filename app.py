# This block imports all required core libraries
# Streamlit is used for UI rendering and interactivity
# Pandas and NumPy handle data processing and statistics
# OS, JSON, IO support environment access and exports
# This block imports all required core libraries
# Streamlit is used for UI rendering and interactivity
# Pandas and NumPy handle data processing and statistics
# OS, JSON, IO support environment access and exports
import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os
import json
import io

# This block loads environment variables securely
# dotenv helps keep API keys outside source code
# GenAI SDK enables communication with Gemini models
# This is required before model initialization
from dotenv import load_dotenv
import google.generativeai as genai


# This block defines application-level page settings
# Title appears on browser tab
# Wide layout is selected for dashboard-style UI
# Configuration runs once at app start
st.set_page_config(
    page_title="GenAI Model Recommendation System",
    layout="wide"
)

# This block renders the main header UI
# Gradient background improves visual hierarchy
# HTML styling is injected using Streamlit markdown
# Header explains the system purpose clearly
st.markdown(
    """
    <div style="
        background:linear-gradient(90deg,#020617,#1e3a8a,#4338ca);
        padding:40px;
        border-radius:26px;
        margin-bottom:30px;
    ">
        <h1 style="color:white;font-size:46px;margin:0;">
            🤖 GenAI Model Recommendation System
        </h1>
        <p style="color:#e5e7eb;font-size:20px;margin-top:10px;">
            Model selection + Statistical intelligence for AI/ML engineers
        </p>
    </div>
    """,
    unsafe_allow_html=True
)

# This block loads environment variables from .env
# API key is fetched securely using os.getenv
# App execution is stopped if key is missing
# GenAI client is configured here
load_dotenv()
API_KEY = os.getenv("GOOGLE_API_KEY")

if not API_KEY:
    st.error("GOOGLE_API_KEY not found. Please check your .env file.")
    st.stop()

genai.configure(api_key=API_KEY)
model = genai.GenerativeModel("gemini-2.5-flash-lite")


# This block contains reusable helper functions
# Each function performs a single logical task
# Keeps main Streamlit flow clean and readable
# No UI elements are placed inside helpers
def detect_problem_type(target_series):
    if target_series.dtype.kind in "ifu":
        return "Regression"
    return "Classification"



def compute_confidence(profile):
    score = 100
    if profile["missing_ratio"] > 0.2:
        score -= 15
    if profile["target_imbalance"] > 0.8:
        score -= 20
    if profile["rows"] < 5000:
        score -= 15
    return max(40, score)


def profile_data(df, target):
    numeric_cols = df.select_dtypes(include=np.number)
    vc = df[target].value_counts(normalize=True)

    return {
        "rows": int(df.shape[0]),
        "columns": int(df.shape[1]),
        "numeric_features": int(numeric_cols.shape[1]),
        "categorical_features": int(df.shape[1] - numeric_cols.shape[1]),
        "missing_ratio": round(df.isnull().mean().mean(), 3),
        "avg_skewness": round(numeric_cols.skew().mean(), 3) if numeric_cols.shape[1] > 0 else 0,
        "avg_kurtosis": round(numeric_cols.kurtosis().mean(), 3) if numeric_cols.shape[1] > 0 else 0,
        "target_classes": int(len(vc)),
        "target_imbalance": round(1 - vc.max(), 3)
    }


# This block builds the GenAI prompt dynamically
# Dataset statistics are injected into prompt text
# Strict JSON-only output is enforced
# Enables reliable downstream parsing
def build_prompt(profile):
    return f"""
Return ONLY valid JSON.

{{
  "primary_model": {{
    "name": "",
    "accuracy_range": "",
    "why": []
  }},
  "secondary_model": {{
    "name": "",
    "accuracy_range": "",
    "why": []
  }},
  "why_not_others": [
    {{
      "model": "",
      "reason": ""
    }}
  ],
  "neural_network": {{
    "recommended": true/false,
    "reason": ""
  }},
  "data_risks": []
}}

Dataset profile:
Rows: {profile['rows']}
Columns: {profile['columns']}
Numeric features: {profile['numeric_features']}
Categorical features: {profile['categorical_features']}
Missing ratio: {profile['missing_ratio']}
Target classes: {profile['target_classes']}
Target imbalance: {profile['target_imbalance']}
Skewness: {profile['avg_skewness']}
Kurtosis: {profile['avg_kurtosis']}
"""


# This block safely parses GenAI text output
# Extracts JSON substring from raw text
# Prevents app crash on malformed output
# Returns None if parsing fails
def parse_response(text):
    try:
        return json.loads(text[text.find("{"): text.rfind("}") + 1])
    except:
        return None


# This block generates adaptive EDA recommendations
# Logic depends on missing values, skewness, outliers
# No static rules or hardcoded advice
# Output varies per dataset profile
def suggest_eda_actions(profile, outlier_count, total_rows):
    actions = []

    outlier_ratio = outlier_count / total_rows if total_rows else 0

    if profile["missing_ratio"] > 0.2:
        actions.append(("❗ High missing values",
                        "Use KNN / MICE imputation or drop high-missing features"))
    elif profile["missing_ratio"] > 0.05:
        actions.append(("⚠️ Moderate missing values",
                        "Use median imputation + missing indicator"))
    else:
        actions.append(("✅ Low missing values",
                        "Simple mean / median imputation is sufficient"))

    if outlier_ratio > 0.05:
        actions.append(("❗ Heavy outliers detected",
                        "Apply log transform or prefer tree-based / robust models"))
    elif outlier_ratio > 0.01:
        actions.append(("⚠️ Some outliers detected",
                        "Apply IQR capping (winsorization)"))
    else:
        actions.append(("✅ Minimal outliers",
                        "Outlier removal is safe"))

    if abs(profile["avg_skewness"]) > 1:
        actions.append(("⚠️ High skewness",
                        "Apply log / Box-Cox transformation"))
    elif abs(profile["avg_skewness"]) > 0.5:
        actions.append(("ℹ️ Mild skewness",
                        "Consider sqrt or log transform"))
    else:
        actions.append(("✅ Low skewness",
                        "No transformation required"))

    if profile["avg_kurtosis"] > 3:
        actions.append(("⚠️ Heavy-tailed distribution",
                        "Prefer RobustScaler over StandardScaler"))

    return actions


# This block handles dataset upload from user
# Only CSV files are supported
# Data is loaded into pandas DataFrame
# UI updates once file is successfully read
uploaded_file = st.file_uploader("📂 Upload CSV Dataset", type=["csv"])

if uploaded_file:
    df = pd.read_csv(uploaded_file)
    st.success("Dataset loaded successfully")
    st.markdown("<h1 style='font-size:42px;'>📦 Dataset Summary</h1>", unsafe_allow_html=True)

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Rows", df.shape[0])
    c2.metric("Columns", df.shape[1])
    missing_rows = df.isnull().any(axis=1).sum()
    missing_pct = round(df.isnull().mean().mean() * 100, 2)

    c3.metric("Rows with Missing", missing_rows)
    c4.metric("Missing %", f"{missing_pct}%")

    c4.metric("Memory (MB)", round(df.memory_usage(deep=True).sum() / 1024**2, 2))

    c5, c6 = st.columns(2)
    c5.metric("Numeric Columns", len(df.select_dtypes(include=np.number).columns))
    c6.metric("Categorical Columns", len(df.select_dtypes(exclude=np.number).columns))

    target = st.selectbox("🎯 Select Target Column", df.columns)

    if target:
        profile = profile_data(df, target)
        confidence = compute_confidence(profile)
        problem_type = detect_problem_type(df[target])
        st.markdown("<h1 style='font-size:42px;margin-top:20px;'>🧠 ML Problem Type</h1>", unsafe_allow_html=True)

        st.markdown(
            f"""
            <p style="font-size:36px;font-weight:700;color:#22c55e;">
                {problem_type}
            </p>
            """,
            unsafe_allow_html=True
        )


        tab1, tab2 = st.tabs(["🏆 Model Recommendation", "📊 Statistical Analysis"])

        # This block renders model recommendation tab
        # Confidence score is visualized using progress bar
        # GenAI call is triggered on button click
        # Results are shown in structured layout
        with tab1:
            st.subheader("🔐 Model Confidence")
            st.progress(confidence / 100)
            st.caption(f"{confidence}% confidence")

            if st.button("🔮 Generate Model Recommendation"):
                with st.spinner("GenAI analyzing dataset..."):
                    response = model.generate_content(build_prompt(profile))

                result = parse_response(response.text)

                if not result:
                    st.error("Failed to parse GenAI response")
                else:
                    st.subheader("🏆 Recommended Models")

                    left, right = st.columns(2)

                    with left:
                        st.markdown(
                            f"""
                            <h1 style="color:#22c55e;margin-bottom:6px;">
                                {result['primary_model']['name']}
                            </h1>
                            <p style="color:#9ca3af;">Accuracy Range</p>
                            <h2>{result['primary_model']['accuracy_range']}</h2>
                            """,
                            unsafe_allow_html=True
                        )

                    with right:
                        st.markdown(
                            f"""
                            <h1 style="color:#3b82f6;margin-bottom:6px;">
                                {result['secondary_model']['name']}
                            </h1>
                            <p style="color:#9ca3af;">Accuracy Range</p>
                            <h2>{result['secondary_model']['accuracy_range']}</h2>
                            """,
                            unsafe_allow_html=True
                        )

                    st.subheader("📋 Model Comparison")
                    df_cmp = pd.DataFrame([
                        {
                            "Model": result["primary_model"]["name"],
                            "Role": "Primary",
                            "Accuracy": result["primary_model"]["accuracy_range"]
                        },
                        {
                            "Model": result["secondary_model"]["name"],
                            "Role": "Backup",
                            "Accuracy": result["secondary_model"]["accuracy_range"]
                        }
                    ])
                    st.dataframe(df_cmp, use_container_width=True)

                    with st.expander("✅ Why These Models"):
                        for r in result["primary_model"]["why"]:
                            st.write(f"- {r}")

                    with st.expander("🧠 Neural Network Recommendation"):
                        st.write(f"Recommended: **{result['neural_network']['recommended']}**")
                        st.write(result["neural_network"]["reason"])

                    with st.expander("❌ Other Models Considered"):
                        for m in result["why_not_others"]:
                            st.write(f"**{m['model']}** → {m['reason']}")

                    with st.expander("⚠️ Data Risks"):
                        for r in result["data_risks"]:
                            st.write(f"- {r}")

                    report = {
                        "dataset_profile": profile,
                        "problem_type": problem_type,
                        "confidence_score": confidence,
                        "recommendation": result
                    }

                    st.download_button(
                        "⬇️ Download Report (JSON)",
                        data=json.dumps(report, indent=4),
                        file_name="model_recommendation_report.json",
                        mime="application/json"
                    )

        # This block performs statistical analysis
        # User selects numeric feature dynamically
        # Outliers are detected using IQR method
        # EDA actions are shown contextually
        with tab2:
            st.subheader("📊 Statistical Analysis Toolkit")

    numeric_cols = df.select_dtypes(include=np.number).columns.tolist()

    if len(numeric_cols) == 0:
        st.warning("No numeric columns available for statistical analysis.")
    else:
        col = st.selectbox("Select Numeric Column", numeric_cols)

        series = df[col].dropna()

        q1, q3 = series.quantile([0.25, 0.75])
        iqr = q3 - q1
        outliers = series[(series < q1 - 1.5 * iqr) | (series > q3 + 1.5 * iqr)]

        st.subheader("🔍 Outlier Detection (IQR)")
        st.write(f"Outliers detected: {len(outliers)}")

        st.subheader("📈 Numeric Distribution (Quick View)")
        plot_type = st.selectbox("Select Plot Type", ["Histogram", "Boxplot"])

        fig, ax = plt.subplots(figsize=(4.5, 2.5))

        if plot_type == "Histogram":
            ax.hist(series, bins=30)
            ax.set_title(f"Histogram: {col}", fontsize=10)

        if plot_type == "Boxplot":
            ax.boxplot(series, vert=False)
            ax.set_title(f"Boxplot: {col}", fontsize=10)

        ax.tick_params(labelsize=8)
        st.pyplot(fig)

        st.markdown("<h1>🛠️ Recommended EDA Actions</h1>", unsafe_allow_html=True)

        actions = suggest_eda_actions(profile, len(outliers), len(series))
        for title, rec in actions:
            st.markdown(
                f"""
                <div style="font-size:26px;margin-bottom:18px;">
                    <b>{title}</b><br>
                    👉 {rec}
                </div>
                """,
                unsafe_allow_html=True
            )


# This block renders the footer section
# Appears at the bottom of the application
# Used for author attribution
# Styling kept minimal and clear
st.markdown(
    "<hr><p style='text-align:center;font-size:24px;font-weight:700;'>Developed by Anurag Kumar Singh</p>",
    unsafe_allow_html=True
)
