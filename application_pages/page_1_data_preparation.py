import streamlit as st
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from aif360.datasets import BinaryLabelDataset
from aif360.metrics import ClassificationMetric, BinaryLabelDatasetMetric
from aif360.sklearn.datasets import fetch_german_credit
import io # Added for df.info()

# --- Helper Functions (can be moved to a separate utils.py if desired, but for this task, keeping in page for self-containment) ---

def load_german_credit_data():
    """Loads the German Credit dataset and returns it as a pandas DataFrame."""
    # Using aif360's fetch_german_credit for consistency and ease
    german_dataset = fetch_german_credit(numeric_only=False)
    df = german_dataset.convert_to_dataframe()[0]
    
    # Rename columns to be more Pythonic and avoid spaces
    df.columns = df.columns.str.replace(r'[()-]', '', regex=True).str.replace(' ', '_').str.lower()
    
    return df

def preprocess_data(df):
    """
    Preprocesses the data:
    - Identifies categorical and numerical features.
    - Creates a preprocessing pipeline (StandardScaler for numerical, OneHotEncoder for categorical).
    - Splits data into training and testing sets.
    - Stores preprocessor, feature names, and AIF360 specific group definitions.
    """
    
    # Define target and features
    target_column = 'credit'
    
    # Separate features and target
    X = df.drop(columns=[target_column])
    y = df[target_column]
    
    # Identify protected attributes and define privileged/unprivileged groups for AIF360
    protected_attribute_names_for_aif360_bld = ['sex_male'] # The actual column name after one-hot encoding
    aif360_privileged_groups = [{'sex_male': 1}] # For when sex_male column is 1
    aif360_unprivileged_groups = [{'sex_male': 0}] # For when sex_male column is 0
    
    # Let's define the original categorical and numerical features
    categorical_features = X.select_dtypes(include=['object', 'category']).columns.tolist()
    numerical_features = X.select_dtypes(include=np.number).columns.tolist()
    
    # Preprocessing pipeline
    numerical_transformer = StandardScaler()
    categorical_transformer = OneHotEncoder(handle_unknown='ignore', sparse_output=False) # sparse_output=False for SHAP/LIME
    
    preprocessor = ColumnTransformer(
        transformers=[
            ('num', numerical_transformer, numerical_features),
            ('cat', categorical_transformer, categorical_features)
        ],
        remainder='passthrough' # Keep other columns if any
    )
    
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
    
    # Fit preprocessor on training data
    preprocessor.fit(X_train)
    
    # Get all feature names after preprocessing
    numeric_feature_names_out = numerical_features
    categorical_feature_names_out = preprocessor.named_transformers_['cat'].get_feature_names_out(categorical_features)
    all_feature_names = list(numeric_feature_names_out) + list(categorical_feature_names_out)
    
    # Transform data
    X_train_transformed = preprocessor.transform(X_train)
    X_test_transformed = preprocessor.transform(X_test)
    
    X_train_df = pd.DataFrame(X_train_transformed, columns=all_feature_names, index=X_train.index)
    X_test_df = pd.DataFrame(X_test_transformed, columns=all_feature_names, index=X_test.index)
    
    y_train = y_train.reset_index(drop=True)
    y_test = y_test.reset_index(drop=True)

    
    return X_train_df, X_test_df, y_train, y_test, preprocessor, all_feature_names, \
           protected_attribute_names_for_aif360_bld, aif360_privileged_groups, aif360_unprivileged_groups

def main():
    st.header("Data Preparation")

    st.markdown("""
    This section is dedicated to loading and preparing the dataset for our AI credit decision model.
    Data preprocessing is a crucial step to ensure the quality and suitability of the data for machine learning algorithms.
    """)

    # Initialize session state variables if not already present
    if "df" not in st.session_state:
        st.session_state.df = None
    if "X_train_df" not in st.session_state:
        st.session_state.X_train_df = None
    if "X_test_df" not in st.session_state:
        st.session_state.X_test_df = None
    if "y_train" not in st.session_state:
        st.session_state.y_train = None
    if "y_test" not in st.session_state:
        st.session_state.y_test = None
    if "preprocessor" not in st.session_state:
        st.session_state.preprocessor = None
    if "all_feature_names" not in st.session_state:
        st.session_state.all_feature_names = None
    if "protected_attribute_names" not in st.session_state:
        st.session_state.protected_attribute_names = None
    if "aif360_privileged_groups" not in st.session_state:
        st.session_state.aif360_privileged_groups = None
    if "aif360_unprivileged_groups" not in st.session_state:
        st.session_state.aif360_unprivileged_groups = None

    st.subheader("Load Dataset")

    col1, col2 = st.columns([1, 2])

    with col1:
        st.markdown("Click the button below to load the default `german_credit.csv` dataset. This dataset is commonly used for fairness research and contains various demographic and financial attributes.")
        if st.button("Load Default Dataset (`german_credit.csv`)"):
            with st.spinner("Loading and preprocessing data..."):
                try:
                    df = load_german_credit_data()
                    st.session_state.df = df
                    
                    # Preprocess and store in session state
                    (st.session_state.X_train_df, st.session_state.X_test_df, 
                     st.session_state.y_train, st.session_state.y_test, 
                     st.session_state.preprocessor, st.session_state.all_feature_names, 
                     st.session_state.protected_attribute_names, 
                     st.session_state.aif360_privileged_groups, 
                     st.session_state.aif360_unprivileged_groups) = preprocess_data(df)
                    
                    st.success("Dataset loaded and preprocessed successfully!")
                    st.info(f"Original dataset shape: {df.shape}")
                    st.info(f"Training features shape: {st.session_state.X_train_df.shape}")
                    st.info(f"Test features shape: {st.session_state.X_test_df.shape}")
                    st.info(f"Total features after preprocessing: {len(st.session_state.all_feature_names)}")

                except Exception as e:
                    st.error(f"Error loading or preprocessing data: {e}")
        

    with col2:
        if st.session_state.df is not None:
            st.subheader("Initial Dataset Snapshot (`df.head()`)")
            st.dataframe(st.session_state.df.head())
            
            st.subheader("Dataset Information (`df.info()`, `df.shape`)")
            st.write(f"**Dataset Shape:** {st.session_state.df.shape}")
            st.write("---")
            st.text("Original DataFrame Info:")
            # df.info() prints to stdout, need to capture it or manually display relevant parts
            buffer = io.StringIO()
            st.session_state.df.info(buf=buffer)
            s = buffer.getvalue()
            st.text(s)

            st.markdown("""
            #### Preprocessing Steps Applied:
            1.  **Categorical Encoding:** One-Hot Encoding is applied to all categorical features (e.g., `sex`, `housing`, `purpose`). This converts categorical variables into a numerical format suitable for machine learning models.
            2.  **Numerical Scaling:** Numerical features (e.g., `duration_in_month`, `age_years`) are scaled using `StandardScaler`. This transforms the data to have zero mean and unit variance, preventing features with larger values from dominating the learning process.
            3.  **Data Splitting:** The dataset is split into training (80%) and testing (20%) sets. This ensures that the model is evaluated on unseen data, providing a more reliable measure of its performance.
            4.  **Feature Names:** All feature names after one-hot encoding are preserved for better interpretability in later stages.

            Our target variable is `credit`, where `1` indicates a "Good Credit" and `0` indicates a "Bad Credit" outcome.
            """)
        else:
            st.info("Load the dataset to see its details and preprocessing summary.")