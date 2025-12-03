import streamlit as st
from utils import train_model, evaluate_model

def main():
    st.header("Section 5: Baseline Model Training (Logistic Regression)")
    st.markdown(r"""
    We will train a Logistic Regression model as our baseline classifier. Logistic Regression is a simple, linear model often used for binary classification. It models the probability that a given input belongs to a particular class using the logistic function:
    $$ P(\hat{Y}=1 | X) = \frac{1}{1 + e^{-(\beta_0 + \beta_1X_1 + \dots + \beta_nX_n)}} $$
    where $P(\hat{Y}=1 | X)$ is the probability of the positive class, $X$ are the features, and $\beta$ are the model coefficients.
    """)
    if st.button("Train Baseline Model"):
        if st.session_state.get('X_train_df') is not None:
            with st.spinner("Training baseline model..."):
                st.session_state.baseline_model = train_model(st.session_state.X_train_df, st.session_state.y_train)
                st.success("Baseline Logistic Regression model trained.")
                st.header("Section 6: Baseline Model Evaluation")
                st.markdown("""
                To understand our baseline model's performance, we evaluate it using standard classification metrics:
                *   **Accuracy**: The proportion of correctly classified instances.
                *   **Precision**: The proportion of positive identifications that were actually correct.
                *   **Recall**: The proportion of actual positives that were identified correctly.
                *   **F1-Score**: The harmonic mean of Precision and Recall.
                *   **Confusion Matrix**: A table showing true positives (TP), true negatives (TN), false positives (FP), and false negatives (FN).
                """)
                st.session_state.baseline_accuracy, _, _, _, _, st.session_state.baseline_predictions = \
                    evaluate_model(st.session_state.baseline_model, st.session_state.X_test_df, st.session_state.y_test, "Baseline Model")
        else:
            st.warning("Please load and preprocess data first."))