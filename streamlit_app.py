import streamlit as st
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix

def eda_page():
    st.write("### Exploratory Data Analysis")

    # Load datasets
    datasets = {
        "Chargebacks": "Data/Fraud/chargebacks.csv",
        "Devices": "Data/Fraud/devices.csv",
        "IPs": "Data/Fraud/ips.csv",
        "Merchants": "Data/Fraud/merchants.csv",
        "Transactions": "Data/Fraud/transactions_raw.csv",
        "Users": "Data/Fraud/users.csv"
    }

    dataset_name = st.selectbox("Select Dataset", list(datasets.keys()))
    if dataset_name:
        data_path = datasets[dataset_name]
        df = pd.read_csv(data_path)

        st.write(f"### {dataset_name} Dataset")
        st.dataframe(df.head())

        # Display basic statistics
        st.write("### Basic Statistics")
        st.write(df.describe())

        # Display column information
        st.write("### Column Information")
        st.write(df.info())

        # Plotting
        st.write("### Visualizations")
        if st.checkbox("Show Correlation Heatmap"):
            st.write("#### Correlation Heatmap")
            plt.figure(figsize=(10, 6))
            sns.heatmap(df.corr(), annot=True, cmap="coolwarm")
            st.pyplot(plt)

        if st.checkbox("Show Value Counts for a Column"):
            column = st.selectbox("Select Column", df.columns)
            if column:
                st.bar_chart(df[column].value_counts())

def ml_modeling_page():
    st.write("### Machine Learning Modeling")

    # Load dataset
    dataset_path = "Data/Fraud/transactions_raw.csv"
    df = pd.read_csv(dataset_path)

    st.write("### Transactions Dataset")
    st.dataframe(df.head())

    # Select features and target
    st.write("### Feature Selection")
    all_columns = df.columns.tolist()
    target_column = st.selectbox("Select Target Column", all_columns)
    feature_columns = st.multiselect("Select Feature Columns", [col for col in all_columns if col != target_column])

    if target_column and feature_columns:
        X = df[feature_columns]
        y = df[target_column]

        # Split data
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

        # Train model
        model = RandomForestClassifier()
        model.fit(X_train, y_train)

        # Evaluate model
        y_pred = model.predict(X_test)
        st.write("### Model Evaluation")
        st.write("#### Classification Report")
        st.text(classification_report(y_test, y_pred))

        st.write("#### Confusion Matrix")
        st.write(confusion_matrix(y_test, y_pred))

def main():
    st.title("Fraud Detection Dashboard")

    # Sidebar navigation
    menu = ["EDA", "ML Modeling"]
    choice = st.sidebar.selectbox("Select Page", menu)

    if choice == "EDA":
        eda_page()
    elif choice == "ML Modeling":
        ml_modeling_page()

if __name__ == "__main__":
    main()