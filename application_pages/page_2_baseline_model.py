import streamlit as st
import pandas as pd
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix, classification_report
import matplotlib.pyplot as plt
import seaborn as sns

# --- Helper Functions ---

def plot_confusion_matrix(y_true, y_pred, title="Confusion Matrix"):
    """Plots a confusion matrix."""
    cm = confusion_matrix(y_true, y_pred)
    fig, ax = plt.subplots(figsize=(6, 5))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=ax,
                xticklabels=['Bad Credit (0)', 'Good Credit (1)'] Bedeutschung 'Bad Credit' und 'Good Credit',
                yticklabels=['Bad Credit (0)', 'Good Credit (1)'] Bedeutschung 'Bad Credit' und 'Good Credit'])
    ax.set_xlabel("Predicted Label")
    ax.set_ylabel("True Label")
    ax.set_title(title)
    return fig

def main():
    st.header("Baseline Model Training and Evaluation")

    st.markdown("""
    In this section, we will train a **Logistic Regression** model as our baseline.
    Logistic Regression is a fundamental classification algorithm, often used for binary outcomes.
    It models the probability of a certain class or event occurring.

    #### Mathematical Basis of Logistic Regression:
    The core of Logistic Regression is the sigmoid function, which squashes any real-valued number into a probability between 0 and 1.
    The probability $P(Y=1|X)$ that the dependent variable $Y$ is 1 given input features $X$ is modeled as:

    $$ P(Y=1|X) = \frac{1}{1 + e^{-(\beta_0 + \beta_1X_1 + ... + \beta_nX_n)}} $$

    Where:
    *   $P(Y=1|X)$ is the probability of the positive class (e.g., good credit).
    *   $e$ is the base of the natural logarithm.
    *   $\beta_0$ is the intercept.
    *   $\beta_i$ are the coefficients for each feature $X_i$.

    The model then predicts the class based on a threshold (typically 0.5): if $P(Y=1|X) \ge 0.5$, predict 1; otherwise, predict 0.
    """)

    # Ensure necessary data is in session state
    if "X_train_df" not in st.session_state or "y_train" not in st.session_state or \
       "X_test_df" not in st.session_state or "y_test" not in st.session_state:
        st.warning("Please go to 'Data Preparation' and load/preprocess the data first.")
        return

    st.subheader("Train Baseline Logistic Regression Model")

    if st.button("Train Baseline Model"):
        with st.spinner("Training Logistic Regression model..."):
            try:
                # Initialize and train the model
                baseline_model = LogisticRegression(random_state=42, solver='liblinear') # liblinear for small datasets
                baseline_model.fit(st.session_state.X_train_df, st.session_state.y_train)
                st.session_state.baseline_model = baseline_model

                # Make predictions
                y_pred = baseline_model.predict(st.session_state.X_test_df)
                y_proba = baseline_model.predict_proba(st.session_state.X_test_df)[:, 1]
                
                st.session_state.baseline_predictions = y_pred
                st.session_state.baseline_probabilities = y_proba

                st.success("Baseline Logistic Regression model trained successfully!")

                st.subheader("Model Performance on Test Set")

                # Calculate metrics
                accuracy = accuracy_score(st.session_state.y_test, y_pred)
                precision = precision_score(st.session_state.y_test, y_pred)
                recall = recall_score(st.session_state.y_test, y_pred)
                f1 = f1_score(st.session_state.y_test, y_pred)

                st.session_state.baseline_accuracy = accuracy
                st.session_state.baseline_precision = precision
                st.session_state.baseline_recall = recall
                st.session_state.baseline_f1 = f1

                col1, col2, col3, col4 = st.columns(4)
                col1.metric("Accuracy", f"{accuracy:.4f}")
                col2.metric("Precision", f"{precision:.4f}")
                col3.metric("Recall", f"{recall:.4f}")
                col4.metric("F1-Score", f"{f1:.4f}")

                st.markdown("""
                *   **Accuracy:** The proportion of correctly classified instances out of the total instances.
                *   **Precision:** The proportion of positive identifications that were actually correct. Useful when the cost of false positives is high.
                *   **Recall:** The proportion of actual positives that were identified correctly. Useful when the cost of false negatives is high.
                *   **F1-Score:** The harmonic mean of Precision and Recall, providing a single metric that balances both.
                """)

                st.subheader("Confusion Matrix")
                fig_cm = plot_confusion_matrix(st.session_state.y_test, y_pred, "Baseline Model Confusion Matrix")
                st.pyplot(fig_cm)
                plt.close(fig_cm)

                st.subheader("Classification Report")
                st.text(classification_report(st.session_state.y_test, y_pred))

            except Exception as e:
                st.error(f"Error training or evaluating model: {e}")
    else:
        if "baseline_model" in st.session_state and st.session_state.baseline_model is not None:
            st.info("Baseline model is already trained. Click 'Train Baseline Model' to retrain.")
            st.subheader("Current Baseline Model Performance")
            col1, col2, col3, col4 = st.columns(4)
            col1.metric("Accuracy", f"{st.session_state.baseline_accuracy:.4f}")
            col2.metric("Precision", f"{st.session_state.baseline_precision:.4f}")
            col3.metric("Recall", f"{st.session_state.baseline_recall:.4f}")
            col4.metric("F1-Score", f"{st.session_state.baseline_f1:.4f}")
            
            fig_cm = plot_confusion_matrix(st.session_state.y_test, st.session_state.baseline_predictions, "Baseline Model Confusion Matrix")
            st.pyplot(fig_cm)
            plt.close(fig_cm)
            st.text(classification_report(st.session_state.y_test, st.session_state.baseline_predictions))
        else:
            st.info("Click the button above to train the baseline Logistic Regression model.")