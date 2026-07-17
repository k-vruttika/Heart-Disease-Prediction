import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report

# --------------------------------------------------
# Page configuration
# --------------------------------------------------
st.set_page_config(page_title="Heart Disease Prediction", layout="wide")

st.title("Heart Disease Prediction using Decision Tree")
st.subheader("Data Science Open-Ended Lab Project")

# --------------------------------------------------
# Sidebar navigation
# --------------------------------------------------
st.sidebar.title("Navigation")
section = st.sidebar.radio(
    "Go to",
    ["Home", "Dataset View", "EDA", "Model Training", "Prediction"]
)

# --------------------------------------------------
# Load dataset
# --------------------------------------------------
@st.cache_data
def load_data():
    return pd.read_csv("heart.csv")

df = load_data()

# --------------------------------------------------
# HOME
# --------------------------------------------------
if section == "Home":
    st.markdown("""
    ### Welcome to the Heart Disease Prediction System

    This project uses a **Decision Tree Classifier** to predict
    the presence of heart disease using clinical attributes.

    **Modules included:**
    - Dataset View
    - Exploratory Data Analysis (EDA)
    - Model Training
    - Prediction

    **Team Members:**  
    K Vruttika (RVCE25MCA009)  
    Karthika B (RVCE25MCA057)
    """)

# --------------------------------------------------
# DATASET VIEW
# --------------------------------------------------
elif section == "Dataset View":
    st.header("Dataset Overview")

    st.subheader("Sample Records")
    st.dataframe(df.head(10))

    st.subheader("Dataset Shape")
    st.write(df.shape)

    st.subheader("Missing Values")
    st.write(df.isnull().sum())

    st.subheader("Statistical Summary")
    st.write(df.describe())

# --------------------------------------------------
# EDA
# --------------------------------------------------
elif section == "EDA":
    st.header("Exploratory Data Analysis")

    st.subheader("Target Variable Distribution")
    fig, ax = plt.subplots()
    sns.countplot(x="target", data=df, ax=ax)
    ax.set_title("0 = No Disease, 1 = Disease")
    st.pyplot(fig)

    st.subheader("Correlation Heatmap")
    fig, ax = plt.subplots(figsize=(10, 8))
    sns.heatmap(df.corr(), annot=True, cmap="coolwarm", ax=ax)
    st.pyplot(fig)

    st.subheader("Age Distribution")
    fig, ax = plt.subplots()
    sns.histplot(df["age"], bins=20, kde=True, ax=ax)
    st.pyplot(fig)

    st.subheader("Cholesterol vs Maximum Heart Rate")
    fig, ax = plt.subplots()
    sns.scatterplot(
        x="chol", y="thalach", hue="target", data=df, ax=ax
    )
    st.pyplot(fig)

# --------------------------------------------------
# MODEL TRAINING
# --------------------------------------------------
elif section == "Model Training":
    st.header("Model Training using Decision Tree")

    X = df.drop("target", axis=1)
    y = df["target"]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    # 🔹 NEW: Criterion selection
    criterion = st.selectbox(
        "Select Splitting Criterion",
        ("gini", "entropy")
    )

    max_depth = st.slider(
        "Select Max Depth of Decision Tree",
        1, 20, 5
    )

    if st.button("Train Model"):
        model = DecisionTreeClassifier(
            criterion=criterion,
            max_depth=max_depth,
            random_state=42
        )
        model.fit(X_train, y_train)

        y_pred = model.predict(X_test)
        acc = accuracy_score(y_test, y_pred)

        st.success(
            f"Model trained using {criterion.upper()} criterion | Accuracy: {acc:.2f}"
        )

        # Confusion Matrix
        st.subheader("Confusion Matrix")
        cm = confusion_matrix(y_test, y_pred)
        fig, ax = plt.subplots()
        sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", ax=ax)
        st.pyplot(fig)

        # Classification Report
        st.subheader("Classification Report")
        report_df = pd.DataFrame(
            classification_report(y_test, y_pred, output_dict=True)
        ).transpose()
        st.dataframe(report_df)

        # Decision Tree Visualization
        st.subheader("Decision Tree Visualization")
        fig, ax = plt.subplots(figsize=(20, 10))
        plot_tree(
            model,
            feature_names=X.columns,
            class_names=["No Disease", "Disease"],
            filled=True
        )
        st.pyplot(fig)

        # Save model in session
        st.session_state.model = model
        st.session_state.columns = X.columns

# --------------------------------------------------
# PREDICTION
# --------------------------------------------------
elif section == "Prediction":
    st.header("Heart Disease Prediction")

    if "model" not in st.session_state:
        st.warning("Please train the model first in the 'Model Training' section.")
    else:
        model = st.session_state.model
        cols = st.session_state.columns

        st.subheader("Enter Patient Details")
        input_data = {}

        col1, col2 = st.columns(2)
        for i, col in enumerate(cols):
            with col1 if i % 2 == 0 else col2:
                input_data[col] = st.number_input(
                    col,
                    min_value=float(df[col].min()),
                    max_value=float(df[col].max()),
                    value=float(df[col].mean())
                )

        if st.button("Predict"):
            input_df = pd.DataFrame([input_data])
            prediction = model.predict(input_df)[0]
            probability = model.predict_proba(input_df)[0][1]

            if prediction == 1:
                st.error(f"Disease Detected (Probability: {probability:.2f})")
            else:
                st.success(f"No Disease Detected (Probability: {probability:.2f})")

