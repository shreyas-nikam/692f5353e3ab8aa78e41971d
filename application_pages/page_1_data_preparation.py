import streamlit as st
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
import io  # Added for df.info()

# --- Helper Functions (can be moved to a separate utils.py if desired, but for this task, keeping in page for self-containment) ---


def load_german_credit_data():

    # Fallback: Load German Credit dataset from UCI repository directly
    import urllib.request

    url = "https://archive.ics.uci.edu/ml/machine-learning-databases/statlog/german/german.data"

    # Column names for German Credit dataset
    column_names = [
        'status', 'duration', 'credit_history', 'purpose', 'credit_amount',
        'savings', 'employment', 'installment_rate', 'sex', 'other_debtors',
        'residence_since', 'property', 'age', 'other_installment_plans',
        'housing', 'number_credits', 'job', 'people_liable', 'telephone',
        'foreign_worker', 'credit'
    ]

    try:
        # Download and read the data
        with urllib.request.urlopen(url, timeout=10) as response:
            df = pd.read_csv(io.BytesIO(response.read()),
                             sep=' ',
                             names=column_names,
                             header=None)

        # Convert target: 1=good (1), 2=bad (0)
        df['credit'] = df['credit'].map({1: 1, 2: 0})

        # Map sex column (A91=male, A92=female, etc.)
        df['sex'] = df['sex'].map({
            'A91': 'male', 'A92': 'female', 'A93': 'male', 'A94': 'male', 'A95': 'female'
        })

    except Exception as e2:
        st.error(f"Could not download from UCI either: {str(e2)}")
        st.info("Loading from backup synthetic data...")

        # Last resort: Create a small synthetic dataset for demonstration
        np.random.seed(42)
        n_samples = 1000

        df = pd.DataFrame({
            'status': np.random.choice(['A11', 'A12', 'A13', 'A14'], n_samples),
            'duration': np.random.randint(6, 72, n_samples),
            'credit_history': np.random.choice(['A30', 'A31', 'A32', 'A33', 'A34'], n_samples),
            'purpose': np.random.choice(['A40', 'A41', 'A42', 'A43', 'A44', 'A45', 'A46', 'A48', 'A49', 'A410'], n_samples),
            'credit_amount': np.random.randint(250, 18424, n_samples),
            'savings': np.random.choice(['A61', 'A62', 'A63', 'A64', 'A65'], n_samples),
            'employment': np.random.choice(['A71', 'A72', 'A73', 'A74', 'A75'], n_samples),
            'installment_rate': np.random.randint(1, 5, n_samples),
            'sex': np.random.choice(['male', 'female'], n_samples, p=[0.69, 0.31]),
            'other_debtors': np.random.choice(['A101', 'A102', 'A103'], n_samples),
            'residence_since': np.random.randint(1, 5, n_samples),
            'property': np.random.choice(['A121', 'A122', 'A123', 'A124'], n_samples),
            'age': np.random.randint(19, 75, n_samples),
            'other_installment_plans': np.random.choice(['A141', 'A142', 'A143'], n_samples),
            'housing': np.random.choice(['A151', 'A152', 'A153'], n_samples),
            'number_credits': np.random.randint(1, 5, n_samples),
            'job': np.random.choice(['A171', 'A172', 'A173', 'A174'], n_samples),
            'people_liable': np.random.randint(1, 3, n_samples),
            'telephone': np.random.choice(['A191', 'A192'], n_samples),
            'foreign_worker': np.random.choice(['A201', 'A202'], n_samples),
            'credit': np.random.choice([0, 1], n_samples, p=[0.3, 0.7])
        })

    # Rename columns to be more Pythonic and avoid spaces
    df.columns = df.columns.str.replace(
        r'[()-]', '', regex=True).str.replace(' ', '_').str.lower()

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
    # The actual column name after one-hot encoding
    protected_attribute_names_for_aif360_bld = ['sex_male']
    # For when sex_male column is 1
    aif360_privileged_groups = [{'sex_male': 1}]
    # For when sex_male column is 0
    aif360_unprivileged_groups = [{'sex_male': 0}]

    # Let's define the original categorical and numerical features
    categorical_features = X.select_dtypes(
        include=['object', 'category']).columns.tolist()
    numerical_features = X.select_dtypes(include=np.number).columns.tolist()

    # Preprocessing pipeline
    numerical_transformer = StandardScaler()
    categorical_transformer = OneHotEncoder(
        # sparse_output=False for SHAP/LIME
        handle_unknown='ignore', sparse_output=False)

    preprocessor = ColumnTransformer(
        transformers=[
            ('num', numerical_transformer, numerical_features),
            ('cat', categorical_transformer, categorical_features)
        ],
        remainder='passthrough'  # Keep other columns if any
    )

    # Split data
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y)

    # Fit preprocessor on training data
    preprocessor.fit(X_train)

    # Get all feature names after preprocessing
    numeric_feature_names_out = numerical_features
    categorical_feature_names_out = preprocessor.named_transformers_[
        'cat'].get_feature_names_out(categorical_features)
    all_feature_names = list(numeric_feature_names_out) + \
        list(categorical_feature_names_out)

    # Transform data
    X_train_transformed = preprocessor.transform(X_train)
    X_test_transformed = preprocessor.transform(X_test)

    X_train_df = pd.DataFrame(
        X_train_transformed, columns=all_feature_names, index=X_train.index)
    X_test_df = pd.DataFrame(
        X_test_transformed, columns=all_feature_names, index=X_test.index)

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
                    st.info(
                        f"Training features shape: {st.session_state.X_train_df.shape}")
                    st.info(
                        f"Test features shape: {st.session_state.X_test_df.shape}")
                    st.info(
                        f"Total features after preprocessing: {len(st.session_state.all_feature_names)}")

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
