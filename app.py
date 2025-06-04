import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# ------------------------ SIDEBAR & INFO ------------------------
st.sidebar.title("Cyber Security with Federated Learning & Explainable AI")
st.sidebar.markdown("""
#### By Atik Shahariyar Hasan  
Incoming CS Freshman @ NYU Tandon | Researcher | Innovator

Hi! I'm Atik — an incoming Computer Science student at NYU Tandon, passionate about impactful technology. I’ve led AI/ML research, won global awards, and mentored 500+ rural students in robotics and programming. From explainable AI to quantum computing for healthcare, I’m building the future — one project at a time.
""")

st.sidebar.markdown("---")
st.sidebar.header("Choose Data")
data_choice = st.sidebar.radio(
    "Select your dataset:",
    ("Use sample data", "Upload your own CSV")
)

st.sidebar.header("Model Options")
model_option = st.sidebar.radio(
    "Choose federated learning mode:",
    ("Train from scratch", "Use pre-trained model (fast)")
)

st.sidebar.markdown("---")
st.sidebar.info("Developed by Atik Shahariyar Hasan\n\n[LinkedIn](https://www.linkedin.com/) | [NYU Tandon](https://engineering.nyu.edu/)")

# ------------------------ DATA LOAD SECTION ------------------------

@st.cache_data
def load_sample_data():
    url = "https://drive.google.com/uc?id=1ZUSX0T2PJS5aJE8cDs0x6ykfZHXsPprD"
    return pd.read_csv(url)

def load_user_data(uploaded_file):
    return pd.read_csv(uploaded_file)

if data_choice == "Use sample data":
    try:
        df = load_sample_data()
        st.success("Sample data loaded from Google Drive.")
    except Exception as e:
        st.error(f"Error loading data: {e}")
        st.stop()
else:
    uploaded_file = st.file_uploader("Upload your CSV file", type=["csv"])
    if uploaded_file:
        df = load_user_data(uploaded_file)
        st.success("Uploaded data loaded.")
    else:
        st.warning("Please upload a CSV file.")
        st.stop()

# ------------------------ DATA OVERVIEW ------------------------
st.header("1️⃣ Data Overview")
st.write("**Shape:**", df.shape)
st.write("**Columns:**", list(df.columns))
st.write(df.head())

# ------------------------ MISSING VALUE HANDLING ------------------------
st.header("2️⃣ Data Cleaning")
st.write("**Missing Values:**")
st.write(df.isnull().sum())

if st.button("Fill Missing Values"):
    # Fill numeric with median, categorical with mode
    for col in df.columns:
        if df[col].isnull().any():
            if pd.api.types.is_numeric_dtype(df[col]):
                df[col].fillna(df[col].median(), inplace=True)
            else:
                df[col].fillna(df[col].mode()[0], inplace=True)
    st.success("Missing values filled.")
    st.write(df.isnull().sum())

# ------------------------ EDA VISUALIZATION ------------------------
st.header("3️⃣ Attack Type Analysis (EDA)")

if "Attack" in df.columns:
    attack_counts = df['Attack'].value_counts()
    fig1, ax1 = plt.subplots()
    ax1.bar(attack_counts.index, attack_counts.values)
    plt.xticks(rotation=65)
    st.pyplot(fig1)

    fig2, ax2 = plt.subplots()
    ax2.pie(attack_counts.values, labels=attack_counts.index, autopct="%1.1f%%")
    st.pyplot(fig2)
else:
    st.warning("'Attack' column not found in your data.")

# ------------------------ FEDERATED LEARNING & PREDICTION ------------------------
st.header("4️⃣ Federated Learning & Prediction")

st.write("Choose to train a model or use a pre-trained model for prediction.")

# Placeholder functions — replace with your real model/XAI code!
def train_federated_model(df):
    # Your federated learning training code here!
    st.info("Training model... (placeholder logic)")
    return np.random.randint(0, 2, size=len(df))  # Example: fake predictions

def use_pretrained_model(df):
    st.info("Loading pre-trained model... (placeholder logic)")
    return np.random.randint(0, 2, size=len(df))

if st.button("Run Model"):
    if model_option == "Train from scratch":
        predictions = train_federated_model(df)
    else:
        predictions = use_pretrained_model(df)
    st.success("Prediction complete!")
    df["Prediction"] = predictions
    st.dataframe(df.head())

    csv = df.to_csv(index=False).encode()
    st.download_button("Download Results as CSV", data=csv, file_name="predictions.csv")

# ------------------------ EXPLAINABLE AI (XAI) ------------------------
st.header("5️⃣ Explainable AI (XAI)")

if st.button("Show Feature Importance (XAI)"):
    st.info("Feature importance/SHAP plots go here (replace with your code).")
    fig, ax = plt.subplots()
    ax.bar(["feat1","feat2","feat3"], [0.3,0.5,0.2])
    st.pyplot(fig)

st.info("Thank you for using this demo! For feedback or collaboration, connect with Atik on LinkedIn.")









