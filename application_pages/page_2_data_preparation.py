import streamlit as st
import pandas as pd
from sklearn.model_selection import train_test_split
from utils import load_and_explore_data, preprocess_data
import matplotlib.pyplot as plt

def main():
    st.header("Section 3: Dataset Loading and Initial Exploration")
    st.markdown("""
    We will use a modified version of the 'UCI German Credit Data' dataset. This dataset contains demographic and financial attributes that influence credit decisions. It has a binary target variable indicating 'good' or 'bad' credit risk, which we will map to 'loan approval' or 'loan denial'. We will define 'Gender' and 'Age Group' as protected attributes for fairness analysis.
    The dataset is assumed to be available as `german_credit.csv` in the same directory as the notebook.
    """)

    if st.button("Load and Preprocess Data"):
        with st.spinner("Loading and preprocessing data..."):
            st.session_state.df = load_and_explore_data('german_credit.csv')
            X_raw, y, preprocessor, categorical_features, numerical_features, original_cols = preprocess_data(st.session_state.df)
            X_train_pre, X_test_pre, y_train, y_test = train_test_split(X_raw, y, test_size=0.3, random_state=42, stratify=y)
            preprocessor.fit(X_train_pre)
            X_train_transformed = preprocessor.transform(X_train_pre)
            X_test_transformed = preprocessor.transform(X_test_pre)
            
            # Dynamically get feature names for one-hot encoded columns
            # Create a list for all feature names, numerical first, then one-hot encoded categorical
            encoded_feature_names = preprocessor.named_transformers_['cat'].get_feature_names_out(categorical_features)
            all_feature_names = list(numerical_features) + list(encoded_feature_names)

            st.session_state.X_train_df = pd.DataFrame(X_train_transformed, columns=all_feature_names, index=X_train_pre.index)
            st.session_state.X_test_df = pd.DataFrame(X_test_transformed, columns=all_feature_names, index=X_test_pre.index)
            st.session_state.y_train = y_train
            st.session_state.y_test = y_test
            st.session_state.preprocessor = preprocessor
            st.session_state.all_feature_names = all_feature_names
            st.session_state.protected_attribute_names = ['Gender_0', 'Gender_1']
            st.session_state.aif360_privileged_groups = [{'Gender_1': 1}]
            st.session_state.aif360_unprivileged_groups = [{'Gender_0': 1}]
            st.success("Data loaded and preprocessed!")

            st.markdown("Dataset Head:")
            st.dataframe(st.session_state.df.head())
            st.markdown(f"Dataset Shape: {st.session_state.df.shape}")
            st.markdown(f"Shape of X_train_df: {st.session_state.X_train_df.shape}")
            st.markdown(f"Shape of X_test_df: {st.session_state.X_test_df.shape}")
            st.markdown(f"Protected attribute names: {st.session_state.protected_attribute_names}")
    else:
        st.info("Click 'Load and Preprocess Data' to begin.")
