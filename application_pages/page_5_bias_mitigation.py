import streamlit as st
import pandas as pd
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix, classification_report
import matplotlib.pyplot as plt
import seaborn as sns
from aif360.datasets import BinaryLabelDataset
from aif360.metrics import ClassificationMetric
from aif360.algorithms.preprocessing import Reweighing

# Re-import helper functions from fairness analysis if needed, or pass them
# For self-containment, I'll include necessary helper functions here too or adapt.
# For plot_confusion_matrix, I'll just redefine it to avoid circular imports.

def plot_confusion_matrix(y_true, y_pred, title="Confusion Matrix"):
    """Plots a confusion matrix."""
    cm = confusion_matrix(y_true, y_pred)
    fig, ax = plt.subplots(figsize=(6, 5))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=ax,
                xticklabels=['Bad Credit (0)', 'Good Credit (1)'],
                yticklabels=['Bad Credit (0)', 'Good Credit (1)'])
    ax.set_xlabel("Predicted Label")
    ax.set_ylabel("True Label")
    ax.set_title(title)
    return fig

def create_aif360_dataset_for_mitigation(X_df, y_series, protected_attribute_names, privileged_groups, unprivileged_groups):
    """
    Creates an AIF360 BinaryLabelDataset from preprocessed data for mitigation algorithms.
    Important: The `protected_attribute_names` list should contain the *actual column names*
    in the `X_df` after preprocessing that represent the protected attributes.
    `privileged_groups` and `unprivileged_groups` must contain dictionaries
    where keys are these `protected_attribute_names` and values are their corresponding
    privileged/unprivileged values (0 or 1 for one-hot encoded).
    """
    df_aif = X_df.copy()
    df_aif['credit'] = y_series.values # Add target column
    
    # Use the keys from the group definitions to derive the actual protected attribute names for AIF360
    actual_protected_attribute_names_for_aif = protected_attribute_names

    bld = BinaryLabelDataset(
        df=df_aif,
        label_names=['credit'],
        protected_attribute_names=actual_protected_attribute_names_for_aif,
        favorable_label=1,
        unfavorable_label=0
    )
    return bld

def calculate_fairness_metrics(dataset_true, dataset_pred, privileged_groups, unprivileged_groups):
    """Calculates Statistical Parity Difference (SPD) and Equal Opportunity Difference (EOD)."""
    
    metric = ClassificationMetric(
        dataset_true,
        dataset_pred,
        unprivileged_groups=unprivileged_groups,
        privileged_groups=privileged_groups
    )
    
    spd = metric.statistical_parity_difference()
    eod = metric.equal_opportunity_difference()
    
    return spd, eod

def main():
    st.header("Bias Mitigation")

    st.markdown("""
    Bias mitigation techniques aim to reduce or eliminate unfairness in machine learning models.
    This section explores two simple, yet effective, mitigation strategies: **Reweighing** and **Classification Threshold Adjustment**.

    ---

    ### 🛠️ Bias Mitigation Strategies

    #### 1. Reweighing (Pre-processing technique)
    **Reweighing** is a pre-processing technique that assigns different weights to individual training examples based on their protected attributes and labels. This rebalancing helps to equalize the representation of different demographic groups in the training data, leading to a fairer model.

    The weights are designed such that the joint probability distribution of the protected attribute and the label becomes fair.
    For example, if a privileged group with a favorable outcome is over-represented, their samples will be down-weighted.

    #### 2. Classification Threshold Adjustment (Post-processing technique)
    **Classification Threshold Adjustment** is a post-processing technique that modifies the classification threshold (typically 0.5 for logistic regression) to achieve fairness goals.
    Instead of using a single threshold for all groups, different thresholds can be applied to different demographic groups to balance fairness and performance.
    For instance, if the unprivileged group has a lower predicted probability for a favorable outcome, its threshold can be lowered to increase their favorable outcome rate.
    """)

    # Ensure necessary data, model, and AIF360 definitions are in session state
    if "X_train_df" not in st.session_state or "y_train" not in st.session_state or \
       "X_test_df" not in st.session_state or "y_test" not in st.session_state or \
       "baseline_model" not in st.session_state or \
       "protected_attribute_names" not in st.session_state or \
       "aif360_privileged_groups" not in st.session_state or \
       "aif360_unprivileged_groups" not in st.session_state:
        st.warning("Please ensure data is loaded/preprocessed and the 'Baseline Model' is trained.")
        return

    # Initialize session state for mitigated models and metrics
    if "reweighed_model" not in st.session_state:
        st.session_state.reweighed_model = None
    if "reweighed_predictions" not in st.session_state:
        st.session_state.reweighed_predictions = None
    if "reweighed_probabilities" not in st.session_state:
        st.session_state.reweighed_probabilities = None
    if "reweighed_accuracy" not in st.session_state:
        st.session_state.reweighed_accuracy = None
    if "reweighed_spd" not in st.session_state:
        st.session_state.reweighed_spd = None
    if "reweighed_eod" not in st.session_state:
        st.session_state.reweighed_eod = None
    
    if "adjusted_accuracy" not in st.session_state:
        st.session_state.adjusted_accuracy = None
    if "adjusted_spd" not in st.session_state:
        st.session_state.adjusted_spd = None
    if "adjusted_eod" not in st.session_state:
        st.session_state.adjusted_eod = None
    if "threshold_adjusted_predictions" not in st.session_state:
        st.session_state.threshold_adjusted_predictions = None

    st.subheader("Apply Reweighing and Retrain Model")

    if st.button("Apply Reweighing & Retrain Model"):
        with st.spinner("Applying Reweighing and retraining model..."):
            try:
                # Create AIF360 dataset for training
                dataset_orig_train = create_aif360_dataset_for_mitigation(
                    st.session_state.X_train_df,
                    st.session_state.y_train,
                    st.session_state.protected_attribute_names,
                    st.session_state.aif360_privileged_groups,
                    st.session_state.aif360_unprivileged_groups
                )

                if dataset_orig_train is None:
                    st.error("Failed to create AIF360 training dataset for Reweighing.")
                    return

                # Reweighing algorithm
                RW = Reweighing(
                    unprivileged_groups=st.session_state.aif360_unprivileged_groups,
                    privileged_groups=st.session_state.aif360_privileged_groups
                )
                dataset_transf_train = RW.fit_transform(dataset_orig_train)

                # Get the sample weights
                sample_weights = dataset_transf_train.instance_weights

                # Retrain Logistic Regression model with sample weights
                reweighed_model = LogisticRegression(random_state=42, solver='liblinear')
                reweighed_model.fit(st.session_state.X_train_df, st.session_state.y_train, sample_weight=sample_weights)
                st.session_state.reweighed_model = reweighed_model

                # Make predictions
                y_pred_reweighed = reweighed_model.predict(st.session_state.X_test_df)
                y_proba_reweighed = reweighed_model.predict_proba(st.session_state.X_test_df)[:, 1]
                
                st.session_state.reweighed_predictions = y_pred_reweighed
                st.session_state.reweighed_probabilities = y_proba_reweighed

                # Evaluate performance
                st.session_state.reweighed_accuracy = accuracy_score(st.session_state.y_test, y_pred_reweighed)

                # Calculate fairness metrics
                dataset_true_test = create_aif360_dataset_for_mitigation(
                    st.session_state.X_test_df, 
                    st.session_state.y_test,
                    st.session_state.protected_attribute_names,
                    st.session_state.aif360_privileged_groups,
                    st.session_state.aif360_unprivileged_groups
                )
                
                dataset_pred_reweighed = dataset_true_test.copy()
                dataset_pred_reweighed.labels = y_pred_reweighed.reshape(-1, 1)
                dataset_pred_reweighed.scores = y_proba_reweighed.reshape(-1, 1)

                st.session_state.reweighed_spd, st.session_state.reweighed_eod = calculate_fairness_metrics(
                    dataset_true_test,
                    dataset_pred_reweighed,
                    st.session_state.aif360_privileged_groups,
                    st.session_state.aif360_unprivileged_groups
                )

                st.success("Reweighing applied and model retrained successfully!")
                st.subheader("Reweighed Model Performance & Fairness")
                col1, col2, col3 = st.columns(3)
                col1.metric("Accuracy", f"{st.session_state.reweighed_accuracy:.4f}")
                col2.metric("SPD", f"{st.session_state.reweighed_spd:.4f}")
                col3.metric("EOD", f"{st.session_state.reweighed_eod:.4f}")

                st.markdown("---")
                st.subheader("Confusion Matrix (Reweighed Model)")
                fig_cm_reweighed = plot_confusion_matrix(st.session_state.y_test, y_pred_reweighed, "Reweighed Model Confusion Matrix")
                st.pyplot(fig_cm_reweighed)
                plt.close(fig_cm_reweighed)

            except Exception as e:
                st.error(f"Error applying Reweighing or retraining model: {e}")
    else:
        if st.session_state.reweighed_model is not None:
            st.info("Reweighed model is already trained.")
            st.subheader("Reweighed Model Performance & Fairness (Pre-calculated)")
            col1, col2, col3 = st.columns(3)
            col1.metric("Accuracy", f"{st.session_state.reweighed_accuracy:.4f}")
            col2.metric("SPD", f"{st.session_state.reweighed_spd:.4f}")
            col3.metric("EOD", f"{st.session_state.reweighed_eod:.4f}")
            
            st.markdown("---")
            st.subheader("Confusion Matrix (Reweighed Model)")
            fig_cm_reweighed = plot_confusion_matrix(st.session_state.y_test, st.session_state.reweighed_predictions, "Reweighed Model Confusion Matrix")
            st.pyplot(fig_cm_reweighed)
            plt.close(fig_cm_reweighed)
        else:
            st.info("Click the button above to apply reweighing and retrain the model.")
            
    st.markdown("---")
    st.subheader("Apply Classification Threshold Adjustment")

    if st.button("Apply Threshold Adjustment & Re-evaluate Model"):
        with st.spinner("Applying threshold adjustment..."):
            try:
                if "baseline_model" not in st.session_state or "baseline_probabilities" not in st.session_state:
                    st.error("Baseline model and its probabilities are required for threshold adjustment. Please train the baseline model first.")
                    return

                # Predictions from the baseline model are used for threshold adjustment
                y_proba_baseline = st.session_state.baseline_probabilities

                # Create AIF360 dataset for true labels for evaluation
                dataset_true_test = create_aif360_dataset_for_mitigation(
                    st.session_state.X_test_df, 
                    st.session_state.y_test,
                    st.session_state.protected_attribute_names,
                    st.session_state.aif360_privileged_groups,
                    st.session_state.aif360_unprivileged_groups
                )
                
                if dataset_true_test is None:
                    st.error("Failed to create AIF360 test dataset for threshold adjustment evaluation.")
                    return
                
                # Assign baseline probabilities to the AIF360 dataset for threshold optimization
                dataset_pred_optim = dataset_true_test.copy()
                dataset_pred_optim.scores = y_proba_baseline.reshape(-1, 1)

                
                best_threshold = 0.5 # Start with default
                best_spd_abs = abs(st.session_state.baseline_spd) if st.session_state.baseline_spd is not None else 1.0
                
                # Search for a better threshold for the unprivileged group if SPD is negative (privileged favored)
                # Or for the privileged group if SPD is positive (unprivileged favored)
                
                # This is a very basic grid search for illustration.
                thresholds = np.linspace(0.01, 0.99, 50)
                
                for thresh in thresholds:
                    # Apply threshold to predict for the specific group or globally to see effect
                    # For simplicity, let's try a global threshold optimization here for a better SPD/EOD.
                    y_pred_adjusted_temp = (y_proba_baseline >= thresh).astype(int)
                    
                    dataset_pred_temp = dataset_true_test.copy()
                    dataset_pred_temp.labels = y_pred_adjusted_temp.reshape(-1, 1)
                    
                    temp_spd, _ = calculate_fairness_metrics(
                        dataset_true_test,
                        dataset_pred_temp,
                        st.session_state.aif360_privileged_groups,
                        st.session_state.aif360_unprivileged_groups
                    )
                    
                    if abs(temp_spd) < best_spd_abs:
                        best_spd_abs = abs(temp_spd)
                        best_threshold = thresh

                st.info(f"Optimized global threshold found: {best_threshold:.4f} (from baseline probabilities)")

                # Apply the best threshold to get adjusted predictions
                y_pred_adjusted = (y_proba_baseline >= best_threshold).astype(int)
                st.session_state.threshold_adjusted_predictions = y_pred_adjusted

                # Evaluate performance and fairness with adjusted predictions
                st.session_state.adjusted_accuracy = accuracy_score(st.session_state.y_test, y_pred_adjusted)
                
                dataset_pred_adjusted = dataset_true_test.copy()
                dataset_pred_adjusted.labels = y_pred_adjusted.reshape(-1, 1)
                dataset_pred_adjusted.scores = y_proba_baseline.reshape(-1, 1) # Still use original probabilities as scores

                st.session_state.adjusted_spd, st.session_state.adjusted_eod = calculate_fairness_metrics(
                    dataset_true_test,
                    dataset_pred_adjusted,
                    st.session_state.aif360_privileged_groups,
                    st.session_state.aif360_unprivileged_groups
                )

                st.success(f"Threshold adjusted to {best_threshold:.4f} and model re-evaluated!")
                st.subheader("Threshold-Adjusted Model Performance & Fairness")
                col1, col2, col3 = st.columns(3)
                col1.metric("Accuracy", f"{st.session_state.adjusted_accuracy:.4f}")
                col2.metric("SPD", f"{st.session_state.adjusted_spd:.4f}")
                col3.metric("EOD", f"{st.session_state.adjusted_eod:.4f}")

                st.markdown("---")
                st.subheader("Confusion Matrix (Threshold-Adjusted Model)")
                fig_cm_adjusted = plot_confusion_matrix(st.session_state.y_test, y_pred_adjusted, "Threshold-Adjusted Model Confusion Matrix")
                st.pyplot(fig_cm_adjusted)
                plt.close(fig_cm_adjusted)


            except Exception as e:
                st.error(f"Error applying threshold adjustment: {e}")
    else:
        if st.session_state.threshold_adjusted_predictions is not None:
            st.info("Threshold adjustment has been applied and re-evaluated.")
            st.subheader("Threshold-Adjusted Model Performance & Fairness (Pre-calculated)")
            col1, col2, col3 = st.columns(3)
            col1.metric("Accuracy", f"{st.session_state.adjusted_accuracy:.4f}")
            col2.metric("SPD", f"{st.session_state.adjusted_spd:.4f}")
            col3.metric("EOD", f"{st.session_state.adjusted_eod:.4f}")
            
            st.markdown("---")
            st.subheader("Confusion Matrix (Threshold-Adjusted Model)")
            fig_cm_adjusted = plot_confusion_matrix(st.session_state.y_test, st.session_state.threshold_adjusted_predictions, "Threshold-Adjusted Model Confusion Matrix")
            st.pyplot(fig_cm_adjusted)
            plt.close(fig_cm_adjusted)
        else:
            st.info("Click the button above to apply threshold adjustment and re-evaluate the baseline model.")