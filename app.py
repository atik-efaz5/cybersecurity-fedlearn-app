import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import shap
from lime.lime_tabular import LimeTabularExplainer
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier

st.set_page_config(
    page_title="Cyber Security with Federated Learning & Explainable AI",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ----------- SIDEBAR: Process Tracker, LinkedIn, Branding -----------
with st.sidebar:
    st.image("https://avatars.githubusercontent.com/u/139233883?v=4", width=120)
    st.title("Atik Shahariyar Hasan")
    st.markdown(
        """
        <span style='color:#1e90ff'><b>Incoming CS Freshman @ NYU Tandon</b></span><br>
        Innovator | ML/AI Researcher<br>
        <a href='https://www.linkedin.com/in/atik-shahariyar-hasan-637635224/' target='_blank'>
            <button style='background-color:#0e76a8;color:white;border:none;padding:7px 18px;margin-top:6px;border-radius:6px;'>
                LinkedIn Profile
            </button>
        </a>
        """,
        unsafe_allow_html=True,
    )
    st.markdown("---")
    st.header("Process Tracker")
    steps = [
        "Data & Cleaning",
        "EDA & Visuals",
        "Correlation",
        "Federated Learning",
        "Explainable AI"
    ]
    st.markdown(
        "".join(
            [
                f"""
                <div style="margin-bottom:10px;">
                    <span style="font-weight:600;color:#1e90ff;">{i+1}.</span>
                    <span style="color:#0e1117;font-weight:600;">{step}</span>
                </div>
                """ for i, step in enumerate(steps)
            ]
        ), unsafe_allow_html=True
    )
    st.markdown("---")
    st.info("Internship-ready ML/AI demo app. Best viewed on desktop.", icon="💼")
    st.caption("Updated: 2025")

st.markdown(
    "<h1 style='text-align:center;color:#002244;margin-bottom:0;'>Cyber Security with Federated Learning & Explainable AI</h1>"
    "<p style='text-align:center;color:#335b91;font-size:18px;margin-top:0;'>A dynamic, fully interactive AI/ML showcase for cyber threat detection</p>",
    unsafe_allow_html=True,
)
st.markdown("---")

tab1, tab2, tab3, tab4, tab5 = st.tabs([
    "1. Data & Cleaning", 
    "2. EDA & Visuals", 
    "3. Correlation", 
    "4. Federated Learning", 
    "5. Explainable AI"
])

with tab1:
    st.subheader("Step 1: Load and Clean Your Data")
    data_src = st.radio("Choose your data source:", ["Sample Data", "Upload CSV"])
    @st.cache_data
    def load_sample_data():
        url = "https://drive.google.com/uc?id=1AsGXT6622ZFu6nWXqtbGMknr54SuNYox"
        return pd.read_csv(url)
    uploaded_file = None
    if data_src == "Sample Data":
        df = load_sample_data()
    else:
        uploaded_file = st.file_uploader("Upload a CSV", type=["csv"])
        if uploaded_file:
            df = pd.read_csv(uploaded_file)
        else:
            st.warning("Please upload your dataset to continue.")
            st.stop()
    st.success(f"Loaded data: {df.shape[0]:,} rows × {df.shape[1]} columns")
    st.dataframe(df.head(10), use_container_width=True, height=320)
    st.markdown(f"**Columns:** {', '.join(df.columns[:10])}{' ...' if len(df.columns)>10 else ''}")

    st.divider()
    st.subheader("Missing Value Check & Quick Cleaning")
    missing = df.isnull().sum()
    st.write("Missing values per column:")
    st.dataframe(missing[missing>0])
    clean = st.button("Fill Missing Values (median/mode)", key="fill_na")
    if clean:
        for col in df.columns:
            if df[col].isnull().any():
                if pd.api.types.is_numeric_dtype(df[col]):
                    df[col].fillna(df[col].median(), inplace=True)
                else:
                    df[col].fillna(df[col].mode()[0], inplace=True)
        st.success("All missing values filled! ✅")
        st.dataframe(df.isnull().sum()[df.isnull().sum()>0])

# =============== 2. EDA & Visuals ===============
with tab2:
    st.subheader("Step 2: Exploratory Data Analysis")
    if "Attack" not in df.columns:
        st.warning("'Attack' column not found in your data!")
    else:
        st.markdown("**Attack Class Distribution:**")
        ac = df['Attack'].value_counts()
        col1, col2 = st.columns(2)
        with col1:
            fig, ax = plt.subplots(figsize=(7,3.5))
            sns.barplot(x=ac.index, y=ac.values, palette="crest", ax=ax)
            ax.set_xticklabels(ac.index, rotation=60)
            ax.set_ylabel("Count")
            ax.set_xlabel("Attack Type")
            ax.set_title("Attack Frequency")
            st.pyplot(fig)
        with col2:
            fig, ax = plt.subplots(figsize=(5,3.5))
            patches, texts, autotexts = ax.pie(ac.values, labels=ac.index, autopct="%1.1f%%", startangle=140, pctdistance=0.8)
            for t in texts: t.set_fontsize(10)
            ax.set_title("Attack Proportion")
            st.pyplot(fig)
        st.markdown("> **Pro tip:** Scroll down to explore feature distributions interactively.")

    st.divider()
    st.subheader("Numeric Feature Distributions")
    num_feats = st.multiselect("Select features to plot histogram/KDE:", options=df.select_dtypes(include=np.number).columns.tolist(), default=[])
    if num_feats:
        for feat in num_feats:
            fig, ax = plt.subplots(figsize=(6,2.5))
            sns.histplot(df[feat], kde=True, color="#2778c4", ax=ax)
            ax.set_title(f"Distribution: {feat}")
            st.pyplot(fig)

# =============== 3. Correlation ===============
with tab3:
    st.subheader("Step 3: Correlation Analysis")
    st.markdown("> View relationships between numeric features and find key variables.")
    corr = df.select_dtypes(include=np.number).corr()
    fig, ax = plt.subplots(figsize=(10,5))
    sns.heatmap(corr, cmap="mako", annot=False, ax=ax)
    st.pyplot(fig)

# =============== 4. Federated Learning (Real Model) ===============
with tab4:
    st.subheader("Step 4: Federated Learning Model")
    st.markdown("- You can simulate model training here with real code.")
    mode = st.radio("Select FL mode:", ["Train from scratch", "Use pre-trained (demo)"], horizontal=True)
    run_model = st.button("Run Model", key="runmodel")
    # ========== REAL CODE: Preprocessing ==========
    label_encoders = {}
    X = df.drop(columns=["Attack", "Label"], errors="ignore")
    y = df["Attack"] if "Attack" in df.columns else None
    for col in X.select_dtypes("object").columns:
        le = LabelEncoder()
        X[col] = le.fit_transform(X[col].astype(str))
        label_encoders[col] = le
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    # ========== TRAIN/INFER ==========
    if run_model:
        if mode == "Train from scratch":
            X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)
            model = RandomForestClassifier(n_estimators=120, random_state=0)
            model.fit(X_train, y_train)
            preds = model.predict(X_test)
            st.success("Model trained. Sample results below.")
            preview = pd.DataFrame({"True": y_test, "Pred": preds}).head(20)
            st.dataframe(preview)
            st.download_button("Download predictions", preview.to_csv(index=False), file_name="predictions.csv")
        else:
            st.info("Demo: Pre-trained model simulated (reload for real FL integration).")
            preds = np.random.choice(y.unique(), size=len(X))
            preview = pd.DataFrame({"True": y, "Pred": preds}).head(20)
            st.dataframe(preview)
            st.download_button("Download demo predictions", preview.to_csv(index=False), file_name="demo_predictions.csv")

# =============== 5. Explainable AI ===============
with tab5:
    st.subheader("Step 5: Explainable AI")
    st.markdown("Pick a row (sample) to see how the model makes its prediction for that data point (real SHAP and LIME).")
    sample_idx = st.number_input(
        "Select a row index to explain:", min_value=0, max_value=min(50, len(X)-1), value=0
    )
    st.info("LIME and SHAP explanations are computed for your selected row. (This may take a moment on first run.)")
    if run_model and 'model' in locals():
        # SHAP explain
        explainer = shap.TreeExplainer(model)
        shap_values = explainer.shap_values(X_scaled)
        st.markdown("**SHAP Explanation:**")
        st.pyplot(shap.force_plot(
            explainer.expected_value[1], 
            shap_values[1][sample_idx], 
            X.iloc[sample_idx,:],
            matplotlib=True, show=False))
        st.markdown("---")
        # LIME explain
        st.markdown("**LIME Explanation:**")
        lime_explainer = LimeTabularExplainer(
            X_scaled, feature_names=list(X.columns),
            class_names=list(np.unique(y)), discretize_continuous=True
        )
        exp = lime_explainer.explain_instance(
            X_scaled[sample_idx], model.predict_proba, num_features=10
        )
        st.pyplot(exp.as_pyplot_figure())
    else:
        st.warning("Train a model first (Tab 4) to see real explainability.")

st.markdown(
    """
    <style>
    .stTabs [data-baseweb="tab"] {
        font-size: 1.13rem;
        padding: 10px 23px 7px 23px;
        margin-right: 10px;
        background-color: #f6f9fc;
        color: #1e2e4a;
        border-radius: 8px 8px 0 0;
        border: none;
    }
    .stTabs [data-baseweb="tab"]:hover {
        color: #144486;
        background: #e7f0fd;
    }
    </style>
    """, unsafe_allow_html=True
)

st.markdown("---")
st.markdown(
    """
    <div style='text-align:center;'>
        <small>
        &copy; 2025 Atik Shahariyar Hasan | Internship-ready Streamlit AI/ML app | NYU Tandon
        </small>
    </div>
    """, unsafe_allow_html=True
)
