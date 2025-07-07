import streamlit as st
import pandas as pd
import sweetviz as sv
import numpy as np
import warnings
import streamlit.components.v1 as components
import os
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.tree import DecisionTreeRegressor, DecisionTreeClassifier
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
from sklearn.metrics import accuracy_score, mean_squared_error
from sklearn.preprocessing import LabelEncoder

# Suppress warnings for cleaner output
warnings.filterwarnings('ignore', category=FutureWarning)
warnings.filterwarnings('ignore', category=UserWarning)
warnings.filterwarnings('ignore', category=DeprecationWarning)

st.set_page_config(
    page_title="ML Model Comparison App",
    page_icon="",
    layout="wide",
    initial_sidebar_state="expanded"
)

st.title(" ML Model Comparison App")
st.markdown(
    "Upload a *CSV* or *Excel (.xlsx)* file, then select your features (X) and target (y) for a tailored Sweetviz report and model comparison.")

# --- File Uploader ---
uploaded_file = st.file_uploader("Choose your data file", type=["csv", "xlsx"])

df = None
if uploaded_file is not None:
    st.success(f"File '{uploaded_file.name}' uploaded successfully!")
    file_type = uploaded_file.name.split('.')[-1].lower()

    try:
        if file_type == "csv":
            df = pd.read_csv(uploaded_file)
        elif file_type == "xlsx":
            df = pd.read_excel(uploaded_file)

        if df is not None and not df.empty:
            st.write("### Data Preview (First 5 rows):")
            st.dataframe(df.head())
            st.write(f"DataFrame Shape: {df.shape[0]} rows, {df.shape[1]} columns")
            st.markdown("---")

            # --- Basic Data Overview Section (for debugging/quick checks) ---
            st.header("Basic Data Overview")
            if st.checkbox("Show DataFrame Info (df.info())"):
                import io
                buffer = io.StringIO()
                df.info(buf=buffer)
                s = buffer.getvalue()
                st.text(s)

            if st.checkbox("Show Descriptive Statistics (df.describe())"):
                st.dataframe(df.describe())

            if st.checkbox("Show Missing Values"):
                missing_data = df.isnull().sum()
                st.write("Number of missing values per column:")
                st.dataframe(missing_data[missing_data > 0])
                if missing_data.sum() == 0:
                    st.info("No missing values found in the DataFrame!")
            st.markdown("---")
            # --- End Basic Data Overview Section ---


            st.header("1. Select Features (X) and Target (y)")

            all_columns = df.columns.tolist()

            # --- Target Variable (y) ---
            selected_target = st.selectbox(
                "Select your *Target Variable (y)*:",
                options=all_columns,
                index=len(all_columns) - 1
            )

            selected_features = st.multiselect(
                "Select your *Feature Columns (X)*:",
                options=[col for col in all_columns if col != selected_target],
                default=[col for col in all_columns if col != selected_target]
            )

            st.markdown("---")
            st.header("2. Generate Sweetviz Report")

            if st.button("Generate Sweetviz Report"):
                if not selected_features:
                    st.warning(
                        "Please select at least some features to generate a meaningful report.")
                else:
                    with st.spinner("Generating Sweetviz report... This might take a moment."):
                        try:
                            report_path = "sweetviz_report.html"
                            my_report = sv.analyze(df, target_feat=selected_target, pairwise_analysis='off')
                            my_report.show_html(report_path, open_browser=False)

                            st.success("Sweetviz report generated!")
                            st.write("### Interactive Report:")

                            with open(report_path, 'r', encoding='utf-8') as f:
                                html_content = f.read()
                            components.html(html_content, height=1000, scrolling=True)

                            st.download_button(
                                label="Download Sweetviz Report (HTML)",
                                data=html_content.encode('utf-8'),
                                file_name="sweetviz_ml_prep_report.html",
                                mime="text/html"
                            )
                            st.info("""
                            The interactive Sweetviz report is displayed above!
                            If you selected a target variable, Sweetviz shows its relationship with all other features.
                            You can also download the report using the button above.
                            """)

                            if os.path.exists(report_path):
                                os.remove(report_path)

                        except Exception as e:
                            st.error(f"An unexpected error occurred while generating the Sweetviz report: {e}")
                            st.exception(e)

            st.header("3. Select ML Model Type")

            model_type = st.selectbox("Select Model Type", ["Regression", "Classification"])

            if model_type == "Regression":
                model_options = ["Linear Regression", "Decision Tree Regressor", "Random Forest Regressor"]
            else:
                model_options = ["Logistic Regression", "Decision Tree Classifier", "Random Forest Classifier"]

            selected_models = st.multiselect("Select Models to Compare", model_options)

            if st.button("Run Models"):
                try:
                    X = df[selected_features]
                    y = df[selected_target]

                    # Encode categorical variables
                    le = LabelEncoder()
                    for col in X.columns:
                        if X[col].dtype == 'object':
                            X[col] = le.fit_transform(X[col].astype(str))

                    if model_type == "Classification":
                        y = le.fit_transform(y.astype(str))

                    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

                    model_results = {}

                    for model_name in selected_models:
                        if model_name == "Linear Regression":
                            model = LinearRegression()
                        elif model_name == "Logistic Regression":
                            model = LogisticRegression(max_iter=1000)
                        elif model_name == "Decision Tree Regressor":
                            model = DecisionTreeRegressor()
                        elif model_name == "Decision Tree Classifier":
                            model = DecisionTreeClassifier()
                        elif model_name == "Random Forest Regressor":
                            model = RandomForestRegressor()
                        elif model_name == "Random Forest Classifier":
                            model = RandomForestClassifier()

                        model.fit(X_train, y_train)
                        y_pred = model.predict(X_test)

                        if model_type == "Regression":
                            score = mean_squared_error(y_test, y_pred)
                            model_results[model_name] = score
                            st.write(f"{model_name} MSE: {score}")
                        else:
                            score = accuracy_score(y_test, np.round(y_pred))
                            model_results[model_name] = score
                            st.write(f"{model_name} Accuracy: {score}")

                    if model_type == "Regression":
                        best_model = min(model_results, key=model_results.get)
                        st.write(f"Best Model: {best_model} with MSE: {model_results[best_model]}")
                    else:
                        best_model = max(model_results, key=model_results.get)
                        st.write(f"Best Model: {best_model} with Accuracy: {model_results[best_model]}")

                except Exception as e:
                    st.error(f"An error occurred while running the models: {e}")
                    st.exception(e)

        else:
            st.warning(
                "The uploaded file resulted in an empty DataFrame or could not be processed. Please check your data.")

    except Exception as e:
        st.error(f"An error occurred while reading or processing the file: {e}")
        st.info("Please ensure your file is a valid CSV or XLSX and try again.")
        st.exception(e)
else:
    st.info("Upload your data file to get started.")

st.markdown("---")
st.markdown("Built with  using Streamlit and Sweetviz.")
