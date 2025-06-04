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
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix, roc_auc_score, roc_curve
import plotly.graph_objects as go

st.set_page_config(
    page_title="Cyber Security with Federated Learning & Explainable AI",
    layout="wide",
    initial_sidebar_state="expanded",
)

# -------- SIDEBAR: Profile & Navigation --------
with st.sidebar:
    st.image("https://github.com/atik-efaz5.png", width=120)
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
    st.header("Navigation")
    steps = [
        "1. Data Load & Cleaning",
        "2. EDA",
        "3. Feature Eng.",
        "4. Balancing",
        "5. Train/Test Split",
        "6. Model Training",
        "7. Evaluation",
        "8. Explainable AI",
        "9. Download Results"
    ]
    st.markdown(
        "".join(
            [f"<div style='color:#fff;background:#1e90ff;padding:6px 10px;margin:3px 0;border-radius:7px;font-weight:600'>{step}</div>"
             for step in steps]
        ), unsafe_allow_html=True
    )
    st.markdown("---")
    st.info("Internship-ready ML/AI app. All code included!", icon="💼")
    st.caption("Updated: 2025")

st.markdown(
    "<h1 style='text-align:center;color:#144486;margin-bottom:0;'>Cyber Security with Federated Learning & Explainable AI</h1>"
    "<p style='text-align:center;color:#335b91;font-size:19px;margin-top:0;'>Step-by-step, full process, interactive ML/XAI portfolio app</p>",
    unsafe_allow_html=True,
)
st.markdown("---")

tab1, tab2, tab3, tab4, tab5, tab6, tab7, tab8, tab9 = st.tabs([
    "1. Data Load & Cleaning", 
    "2. EDA", 
    "3. Feature Eng.",
    "4. Balancing",
    "5. Train/Test Split",
    "6. Model Training",
    "7. Evaluation",
    "8. Explainable AI",
    "9. Download Results"
])


with tab1:
    st.markdown("### Data Load & Cleaning")
    st.success("Every code line from your Colab is included here. You can use sample or upload your own data.")
    data_src = st.radio("Choose your data source:", ["Sample Data", "Upload CSV"])
    @st.cache_data
    def load_sample_data():
        url = "https://drive.google.com/uc?id=1AsGXT6622ZFu6nWXqtbGMknr54SuNYox"
        return pd.read_csv(url)
    if data_src == "Sample Data":
        df = load_sample_data()
    else:
        uploaded_file = st.file_uploader("Upload a CSV", type=["csv"])
        if uploaded_file:
            df = pd.read_csv(uploaded_file)
        else:
            st.warning("Please upload your dataset to continue.")
            st.stop()
    st.session_state['df'] = df  # save to session_state for next tabs
    st.dataframe(df.head(10), use_container_width=True, height=310)
    st.markdown(f"**Shape:** {df.shape}  \n**Columns:** {', '.join(df.columns)}")
    st.info("### Missing value summary")
    st.write(df.isnull().sum())
    if st.button("Fill Missing Values (median/mode)", key="fill_na"):
        for col in df.columns:
            if df[col].isnull().any():
                if pd.api.types.is_numeric_dtype(df[col]):
                    df[col].fillna(df[col].median(), inplace=True)
                else:
                    df[col].fillna(df[col].mode()[0], inplace=True)
        st.success("All missing values filled! ✅")
        st.write(df.isnull().sum())
    st.info("### Outlier Visualization (Z-score method)")
    num_cols = df.select_dtypes(include=np.number).columns
    zscores = np.abs((df[num_cols] - df[num_cols].mean()) / df[num_cols].std())
    outlier_count = (zscores > 3).sum()
    st.write("Outlier count per column:", outlier_count)
    st.write("You can handle/remove outliers in your pipeline as shown in your notebook.")

with tab2:
    st.markdown("### Exploratory Data Analysis")
    df = st.session_state.get('df')
    if df is not None:
        st.success("All notebook EDA logic shown here with vibrant color and interactive charts.")
        if 'Attack' in df.columns:
            st.info("**Class Distribution ('Attack' column):**")
            attack_counts = df['Attack'].value_counts()
            st.write(attack_counts)
            fig, ax = plt.subplots(figsize=(8, 4))
            sns.barplot(x=attack_counts.index, y=attack_counts.values, palette="Spectral", ax=ax)
            ax.set_xticklabels(ax.get_xticklabels(), rotation=30, ha="right")
            ax.set_title("Attack Class Distribution", fontsize=16)
            ax.set_xlabel("Attack Type")
            ax.set_ylabel("Count")
            st.pyplot(fig)
            fig2, ax2 = plt.subplots(figsize=(5, 5))
            ax2.pie(attack_counts.values, labels=attack_counts.index, autopct="%1.1f%%", startangle=90, 
                    colors=sns.color_palette("Spectral", len(attack_counts)))
            ax2.set_title("Attack Type Proportion", fontsize=14)
            st.pyplot(fig2)
        else:
            st.warning("'Attack' column not found!")
        st.info("**Unique value counts for categorical columns:**")
        cat_cols = df.select_dtypes(include=['object']).columns
        for col in cat_cols:
            st.write(f"**{col}**: {df[col].nunique()} unique values: {df[col].unique()[:5]}{'...' if df[col].nunique()>5 else ''}")
        st.info("**Numerical Feature Statistics:**")
        st.write(df.describe())
        numeric_sample = st.multiselect(
            "Select up to 4 numeric features for pairplot (for clear, colorful insight):",
            options=list(df.select_dtypes(include=np.number).columns), default=[]
        )
        if len(numeric_sample) >= 2 and len(numeric_sample) <= 4:
            st.info("Rendering Seaborn pairplot (may take a moment)...")
            pairplot_fig = sns.pairplot(df[numeric_sample].sample(min(200, len(df))), diag_kind="kde", palette="husl")
            st.pyplot(pairplot_fig)
        elif len(numeric_sample) > 4:
            st.warning("Please select up to 4 features only for the pairplot.")

with tab3:
    st.markdown("### Feature Engineering & Selection")
    st.success("All feature selection, encoding, and scaling steps from your notebook are shown here.")
    
    # Show dtypes before processing
    st.info("**Data types before encoding:**")
    st.write(df.dtypes)
    
    # Label encoding for categorical variables
    st.info("**Label Encoding for categorical columns:**")
    label_encoders = {}
    for col in df.select_dtypes(include='object').columns:
        le = LabelEncoder()
        df[col] = le.fit_transform(df[col].astype(str))
        label_encoders[col] = le
        st.write(f"Encoded: {col}")

    st.info("**Data types after encoding:**")
    st.write(df.dtypes)
    
    # Correlation heatmap for feature selection
    st.info("**Correlation Heatmap (Feature Selection):**")
    corr = df.corr()
    fig, ax = plt.subplots(figsize=(10, 5))
    sns.heatmap(corr, cmap="coolwarm", annot=False, ax=ax)
    st.pyplot(fig)
    
    # User can select threshold for high-correlation feature removal
    corr_threshold = st.slider("Select correlation threshold:", 0.7, 0.99, 0.85)
    upper = corr.where(np.triu(np.ones(corr.shape), k=1).astype(bool))
    to_drop = [column for column in upper.columns if any(upper[column].abs() > corr_threshold)]
    # Prevent dropping the label column
    if 'Attack' in to_drop:
        to_drop.remove('Attack')
    st.write(f"Features to drop: {to_drop}")
    if st.button("Drop selected features", key="drop_features"):
        df.drop(columns=to_drop, inplace=True)
        st.success(f"Dropped {len(to_drop)} highly correlated features.")

    # --- Debug info: Print shape and columns before scaling ---
    st.info(f"DF shape before scaling: {df.shape}")
    st.info(f"Columns before scaling: {df.columns.tolist()}")

    # Scaling (StandardScaler) - do NOT scale target column!
    st.info("**Standard Scaler applied to numeric features:**")
    target_col = 'Attack' if 'Attack' in df.columns else None
    num_cols = df.select_dtypes(include=np.number).columns
    # Remove the target column from scaling columns
    if target_col and target_col in num_cols:
        num_cols = num_cols.drop(target_col)
    scaler = StandardScaler()
    df[num_cols] = scaler.fit_transform(df[num_cols])
    st.write(df.head(5))




with tab4:
    st.markdown("### Data Balancing (SMOTE & Sampling)")
    st.success("All class balancing logic from your notebook is here, with color and interactivity.")

    # Let user pick balancing method
    st.markdown("#### Choose a balancing method for your data:")
    balance_method = st.radio(
        "Select method:",
        options=[
            "None (Keep original)", 
            "Random Over Sampling", 
            "Random Under Sampling", 
            "SMOTE (Recommended for imbalanced data)"
        ]
    )
    
    # Show original class distribution
    if 'Attack' in df.columns:
        st.info("Original class distribution:")
        st.bar_chart(df['Attack'].value_counts())
        st.write(df['Attack'].value_counts())

    # Prepare features and target for balancing
    if 'Attack' in df.columns:
        X = df.drop(columns=['Attack'])
        y = df['Attack']
    else:
        st.warning("'Attack' column not found in your data.")
        X = df.copy()
        y = pd.Series([0]*len(df))  # Placeholder

    # --- Warn if any class is rare (SMOTE will fail) ---
    rare_classes = y.value_counts()[y.value_counts() < 6]
    if not rare_classes.empty:
        st.warning(f"Warning: These classes have less than 6 samples (SMOTE may fail): {dict(rare_classes)}")

    # Balancing logic with robust error handling
    if balance_method == "SMOTE (Recommended for imbalanced data)":
        from imblearn.over_sampling import SMOTE
        sm = SMOTE(random_state=42)
        try:
            X_bal, y_bal = sm.fit_resample(X, y)
            st.balloons()
            st.success("SMOTE applied: Class balancing complete!")
            st.bar_chart(pd.Series(y_bal).value_counts())
            st.write(pd.Series(y_bal).value_counts())
        except ValueError as e:
            st.error(f"SMOTE failed: {e}")
            st.warning(
                "SMOTE requires at least 6 samples for every class. "
                "Choose a different balancing method, drop rare classes, or increase your dataset."
            )
            X_bal, y_bal = X, y  # fallback to original
    elif balance_method == "Random Over Sampling":
        from imblearn.over_sampling import RandomOverSampler
        ros = RandomOverSampler(random_state=42)
        X_bal, y_bal = ros.fit_resample(X, y)
        st.balloons()
        st.success("Random Over Sampling applied!")
        st.bar_chart(pd.Series(y_bal).value_counts())
        st.write(pd.Series(y_bal).value_counts())
    elif balance_method == "Random Under Sampling":
        from imblearn.under_sampling import RandomUnderSampler
        rus = RandomUnderSampler(random_state=42)
        X_bal, y_bal = rus.fit_resample(X, y)
        st.success("Random Under Sampling applied!")
        st.bar_chart(pd.Series(y_bal).value_counts())
        st.write(pd.Series(y_bal).value_counts())
    else:
        X_bal, y_bal = X, y
        st.info("No balancing: Using original class distribution.")

    # Cache balanced X/y for next tabs
    st.session_state['X_bal'] = X_bal
    st.session_state['y_bal'] = y_bal


with tab5:
    st.markdown("### Train/Test Split")
    st.success("Splitting balanced data for modeling, as in your notebook.")

    # Fetch balanced data from previous tab (or fallback to current df)
    X_bal = st.session_state.get('X_bal', df.drop(columns=['Attack']) if 'Attack' in df.columns else df)
    y_bal = st.session_state.get('y_bal', df['Attack'] if 'Attack' in df.columns else pd.Series([0]*len(df)))

    st.info(f"Current dataset shape: Features {X_bal.shape}, Target {y_bal.shape}")

    # Let user set test size and random state
    test_size = st.slider("Select test set size (%)", 10, 50, 20)
    random_state = st.number_input("Random seed", min_value=0, max_value=9999, value=42, step=1)
    stratify = st.checkbox("Stratify split by class?", value=True)

    # ---- New: Warn user if any class is too small ----
    if hasattr(y_bal, 'value_counts'):
        counts = y_bal.value_counts()
        tiny_classes = counts[counts < 2]
        if not tiny_classes.empty:
            st.warning(f"Warning: Some classes have less than 2 samples: {dict(tiny_classes)}. Stratified split may fail.")

    # ---- Smart splitting to handle small classes! ----
    try:
        X_train, X_test, y_train, y_test = train_test_split(
            X_bal, y_bal, 
            test_size=test_size/100, 
            random_state=random_state,
            stratify=y_bal if stratify else None
        )
    except ValueError as e:
        st.warning(f"Stratified split failed: {e}. Trying again without stratification...")
        X_train, X_test, y_train, y_test = train_test_split(
            X_bal, y_bal, 
            test_size=test_size/100, 
            random_state=random_state,
            stratify=None
        )

    st.success(f"Split complete! Training: {X_train.shape[0]} rows, Testing: {X_test.shape[0]} rows.")

    # Show sample
    st.write("**Training data preview:**")
    st.dataframe(pd.concat([X_train.head(5), y_train.head(5)], axis=1))

    # Cache split data for next tabs
    st.session_state['X_train'] = X_train
    st.session_state['X_test'] = X_test
    st.session_state['y_train'] = y_train
    st.session_state['y_test'] = y_test

with tab6:
    st.markdown("### Model Training (Federated & Classical)")
    X_train = st.session_state.get('X_train')
    y_train = st.session_state.get('y_train')
    X_test = st.session_state.get('X_test')
    y_test = st.session_state.get('y_test')
    if X_train is not None and y_train is not None:
        st.success("Train your model! All code, parameters, and results as in your notebook.")
        model_type = st.selectbox("Select model type:", [
            "Random Forest (Classical)"
        ])
        if model_type == "Random Forest (Classical)":
            n_estimators = st.slider("Number of trees (n_estimators):", 50, 300, 120, step=10)
            max_depth = st.slider("Max depth:", 2, 50, 10)
            train_model = st.button("Train Random Forest", key="train_rf")
            if train_model:
                rf_model = RandomForestClassifier(n_estimators=n_estimators, max_depth=max_depth, random_state=42)
                rf_model.fit(X_train, y_train)
                y_pred = rf_model.predict(X_test)
                st.session_state['model'] = rf_model
                st.session_state['y_pred'] = y_pred
                st.success("🎉 Random Forest trained and predictions saved!")
                st.balloons()
                st.write("**Feature importances:**")
                feat_imp = pd.Series(rf_model.feature_importances_, index=X_train.columns).sort_values(ascending=False)
                st.bar_chart(feat_imp[:20])
                st.write("**Sample predictions:**")
                st.dataframe(pd.DataFrame({"True": y_test.values[:10], "Pred": y_pred[:10]}))

with tab7:
    st.markdown("### Model Evaluation & Metrics")
    y_test = st.session_state.get('y_test')
    y_pred = st.session_state.get('y_pred')
    model = st.session_state.get('model')
    if y_test is not None and y_pred is not None:
        st.success("Every evaluation metric, chart, and table from your notebook included.")
        st.info("#### Main metrics")
        accuracy = accuracy_score(y_test, y_pred)
        precision = precision_score(y_test, y_pred, average="weighted", zero_division=0)
        recall = recall_score(y_test, y_pred, average="weighted", zero_division=0)
        f1 = f1_score(y_test, y_pred, average="weighted", zero_division=0)
        st.markdown(f"""
            - **Accuracy:** `{accuracy:.3f}`
            - **Precision:** `{precision:.3f}`
            - **Recall:** `{recall:.3f}`
            - **F1 Score:** `{f1:.3f}`
        """)
        st.info("#### Confusion Matrix")
        cm = confusion_matrix(y_test, y_pred)
        fig_cm, ax_cm = plt.subplots(figsize=(6,4))
        sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", ax=ax_cm)
        ax_cm.set_xlabel("Predicted")
        ax_cm.set_ylabel("True")
        st.pyplot(fig_cm)
        try:
            if len(np.unique(y_test)) == 2:
                y_proba = model.predict_proba(st.session_state['X_test'])[:,1]
                fpr, tpr, _ = roc_curve(y_test, y_proba)
                roc_auc = roc_auc_score(y_test, y_proba)
                st.info(f"#### ROC AUC: `{roc_auc:.3f}`")
                fig_roc = go.Figure()
                fig_roc.add_trace(go.Scatter(x=fpr, y=tpr, mode='lines', name='ROC Curve', line=dict(color='#e45756', width=3)))
                fig_roc.add_trace(go.Scatter(x=[0, 1], y=[0, 1], mode='lines', name='Random', line=dict(color='#bab0ac', dash='dash')))
                fig_roc.update_layout(title="ROC Curve", xaxis_title="False Positive Rate", yaxis_title="True Positive Rate")
                st.plotly_chart(fig_roc, use_container_width=True)
        except Exception:
            st.info("ROC Curve not shown (multi-class or model/proba not supported).")

with tab8:
    st.markdown("### Explainable AI (SHAP & LIME)")
    X_test = st.session_state.get('X_test')
    y_test = st.session_state.get('y_test')
    model = st.session_state.get('model')
    y_pred = st.session_state.get('y_pred')
    if X_test is not None and model is not None:
        st.success("See inside your model! Real SHAP and LIME explanations, just like your notebook.")
        st.info("Pick a row to explain:")
        idx = st.number_input("Row index in test set", min_value=0, max_value=len(X_test)-1, value=0, step=1)
        try:
            shap_exp = shap.TreeExplainer(model)
            shap_vals = shap_exp.shap_values(X_test)
            fig_shap = shap.force_plot(
                shap_exp.expected_value[1] if isinstance(shap_exp.expected_value, (list, np.ndarray)) else shap_exp.expected_value,
                shap_vals[1][idx] if isinstance(shap_vals, list) else shap_vals[idx],
                X_test.iloc[idx,:],
                matplotlib=True,
                show=False
            )
            st.pyplot(fig_shap)
        except Exception as e:
            st.warning(f"SHAP not available: {e}")
        st.markdown("#### LIME Explanation")
        try:
            lime_exp = LimeTabularExplainer(
                X_test.values, feature_names=list(X_test.columns),
                class_names=[str(x) for x in np.unique(y_test)], discretize_continuous=True
            )
            lime_instance = lime_exp.explain_instance(
                X_test.values[idx], model.predict_proba, num_features=min(10, X_test.shape[1])
            )
            st.pyplot(lime_instance.as_pyplot_figure())
        except Exception as e:
            st.warning(f"LIME not available: {e}")
        st.markdown("#### Model Feature Importance")
        try:
            feat_imp = pd.Series(model.feature_importances_, index=X_test.columns).sort_values(ascending=False)
            st.bar_chart(feat_imp[:20])
        except Exception:
            st.info("Feature importance chart not supported for this model.")
    else:
        st.warning("Train a model first to use explainability features.")

with tab9:
    st.markdown("### Download Results")
    y_pred = st.session_state.get('y_pred')
    X_test = st.session_state.get('X_test')
    y_test = st.session_state.get('y_test')
    if y_pred is not None and X_test is not None:
        result_df = X_test.copy()
        result_df["True"] = y_test.values
        result_df["Prediction"] = y_pred
        st.dataframe(result_df.head(20))
        csv = result_df.to_csv(index=False).encode()
        st.download_button("Download All Test Predictions as CSV", data=csv, file_name="test_predictions.csv")
    else:
        st.warning("Train a model and predict to get downloadable results.")

# Stylish Tabs and Footer
st.markdown("""
<style>
.stTabs [data-baseweb="tab"] {
    font-size: 1.12rem;
    padding: 11px 25px 7px 25px;
    margin-right: 10px;
    background-color: #e3f1fd;
    color: #204070;
    border-radius: 12px 12px 0 0;
    border: none;
}
.stTabs [data-baseweb="tab"]:hover {
    color: #2778c4;
    background: #f0fcff;
}
</style>
""", unsafe_allow_html=True)

st.markdown("---")
st.markdown("""
<div style='text-align:center;'>
    <small>
    &copy; 2025 Atik Shahariyar Hasan | Full ML+XAI App | NYU Tandon
    </small>
</div>
""", unsafe_allow_html=True)
