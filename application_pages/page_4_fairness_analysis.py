import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from aif360.datasets import BinaryLabelDataset
from aif360.metrics import ClassificationMetric

# --- Helper Functions ---


def create_aif360_dataset(X_df, y_series, protected_attribute_names, privileged_groups, unprivileged_groups, feature_names):
    """
    Creates an AIF360 BinaryLabelDataset from preprocessed data.
    Important: The `protected_attribute_names` list should contain the *actual column names*
    in the `X_df` after preprocessing that represent the protected attributes.
    `privileged_groups` and `unprivileged_groups` must contain dictionaries
    where keys are these `protected_attribute_names` and values are their corresponding
    privileged/unprivileged values (0 or 1 for one-hot encoded).
    """

    # Create a combined DataFrame for AIF360
    df_aif = X_df.copy()
    df_aif['credit'] = y_series.values  # Add target column

    # Use the keys from the group definitions to derive the actual protected attribute names for AIF360
    # Assuming protected_attribute_names from session_state is ['sex_male']
    actual_protected_attribute_names_for_aif = protected_attribute_names

    # Convert to BinaryLabelDataset
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


def plot_fairness_metrics(spd_values, eod_values, labels, title="Fairness Metrics Comparison"):
    """Plots fairness metrics."""

    metrics_df = pd.DataFrame({
        'Model': labels,
        'SPD': spd_values,
        'EOD': eod_values
    }).set_index('Model')

    fig, ax = plt.subplots(1, 2, figsize=(12, 5))

    metrics_df['SPD'].plot(kind='bar', ax=ax[0], color=[
                           'skyblue', 'lightcoral', 'lightgreen'])
    ax[0].axhline(0, color='grey', linestyle='--', linewidth=0.8)
    ax[0].set_title('Statistical Parity Difference (SPD)')
    ax[0].set_ylabel('SPD Value')
    ax[0].tick_params(axis='x', rotation=45)

    metrics_df['EOD'].plot(kind='bar', ax=ax[1], color=[
                           'skyblue', 'lightcoral', 'lightgreen'])
    ax[1].axhline(0, color='grey', linestyle='--', linewidth=0.8)
    ax[1].set_title('Equal Opportunity Difference (EOD)')
    ax[1].set_ylabel('EOD Value')
    ax[1].tick_params(axis='x', rotation=45)

    plt.tight_layout()
    return fig


def main():
    st.header("Fairness Analysis")

    st.markdown("""
    Fairness in AI is about ensuring that models do not discriminate against certain demographic groups.
    This section evaluates the fairness of our baseline model using widely recognized metrics from the `AIF360` library.

    We will focus on the 'sex' attribute (specifically 'sex_male' after one-hot encoding) as our protected attribute for this analysis.

    ---

    ### 📏 Key Fairness Metrics:

    #### 1. Statistical Parity Difference (SPD)
    The **Statistical Parity Difference** measures the difference in the proportion of favorable outcomes (e.g., credit approval) between unprivileged and privileged groups.
    A model achieves statistical parity if the proportion of favorable outcomes is equal across all groups.

    $$ \text{SPD} = P(\hat{Y}=1 \mid \text{unprivileged}) - P(\hat{Y}=1 \mid \text{privileged}) $$

    Where:
    *   $P(\hat{Y}=1 \mid \text{unprivileged})$ is the probability of a favorable prediction for the unprivileged group.
    *   $P(\hat{Y}=1 \mid \text{privileged})$ is the probability of a favorable prediction for the privileged group.

    An SPD value of **0** indicates perfect statistical parity. Values close to 0 (both positive and negative) are generally desired.
    A positive SPD means the unprivileged group receives favorable outcomes more often (which is rare but possible), while a negative SPD means the privileged group receives favorable outcomes more often.

    #### 2. Equal Opportunity Difference (EOD)
    The **Equal Opportunity Difference** measures the difference in true positive rates (recall) between unprivileged and privileged groups. It focuses on whether groups that *should* receive a favorable outcome (true positives) are equally likely to actually receive it.

    $$ \text{EOD} = P(\hat{Y}=1 \mid Y=1, \text{unprivileged}) - P(\hat{Y}=1 \mid Y=1, \text{privileged}) $$

    Where:
    *   $P(\hat{Y}=1 \mid Y=1, \text{unprivileged})$ is the true positive rate for the unprivileged group.
    *   $P(\hat{Y}=1 \mid Y=1, \text{privileged})$ is the true positive rate for the privileged group.

    An EOD value of **0** indicates equal opportunity. Values close to 0 (both positive and negative) are desired.
    A negative EOD suggests that the privileged group has a higher chance of getting a 'Good Credit' prediction when they actually have good credit.
    """)

    # Ensure necessary data, model, and predictions are in session state
    if "X_test_df" not in st.session_state or "y_test" not in st.session_state or \
       "baseline_predictions" not in st.session_state or \
       "all_feature_names" not in st.session_state or \
       "aif360_privileged_groups" not in st.session_state or \
       "aif360_unprivileged_groups" not in st.session_state:
        st.warning(
            "Please ensure data is loaded/preprocessed and the 'Baseline Model' is trained.")
        return

    # Initialize fairness metrics in session state
    if "baseline_spd" not in st.session_state:
        st.session_state.baseline_spd = None
    if "baseline_eod" not in st.session_state:
        st.session_state.baseline_eod = None

    st.subheader("Fairness Analysis for Baseline Model")
    # Displaying directly as 'sex_male' is clearer after OHE
    st.info(f"Analyzing fairness with respect to: `{'sex_male'}`")

    if st.button("Calculate Fairness Metrics for Baseline Model"):
        with st.spinner("Calculating fairness metrics..."):
            try:
                # Create AIF360 datasets
                dataset_true = create_aif360_dataset(
                    st.session_state.X_test_df,
                    st.session_state.y_test,
                    # This is now ['sex_male']
                    st.session_state.protected_attribute_names,
                    st.session_state.aif360_privileged_groups,
                    st.session_state.aif360_unprivileged_groups,
                    st.session_state.all_feature_names
                )

                if dataset_true is None:
                    st.error("Failed to create AIF360 dataset for true labels.")
                    return

                # Create a dataset for predicted labels
                dataset_pred = dataset_true.copy()
                # Reshape to (n, 1)
                dataset_pred.labels = st.session_state.baseline_predictions.reshape(
                    -1, 1)
                # Predicted probabilities
                dataset_pred.scores = st.session_state.baseline_probabilities.reshape(
                    -1, 1)

                # Calculate metrics
                spd, eod = calculate_fairness_metrics(
                    dataset_true,
                    dataset_pred,
                    st.session_state.aif360_privileged_groups,
                    st.session_state.aif360_unprivileged_groups
                )

                st.session_state.baseline_spd = spd
                st.session_state.baseline_eod = eod

                st.success("Fairness metrics calculated successfully!")

                st.subheader("Baseline Model Fairness Metrics")
                st.metric("Statistical Parity Difference (SPD)", f"{spd:.4f}")
                st.metric("Equal Opportunity Difference (EOD)", f"{eod:.4f}")

                fig_fairness = plot_fairness_metrics(
                    [spd], [eod], ['Baseline'], "Baseline Model Fairness Metrics"
                )
                st.pyplot(fig_fairness)
                plt.close(fig_fairness)

                st.markdown("""
                *   **Interpretation of SPD:** If SPD is negative, it indicates that the privileged group has a higher rate of positive predictions than the unprivileged group. A positive SPD means the unprivileged group has a higher rate. The closer to 0, the better.
                *   **Interpretation of EOD:** If EOD is negative, it means the privileged group has a higher true positive rate (recall) than the unprivileged group. A positive EOD means the unprivileged group has a higher true positive rate. The closer to 0, the better.
                """)

            except Exception as e:
                st.error(f"Error calculating fairness metrics: {e}")
    else:
        if st.session_state.baseline_spd is not None and st.session_state.baseline_eod is not None:
            st.info(
                "Fairness metrics for the baseline model have already been calculated.")
            st.subheader("Baseline Model Fairness Metrics (Pre-calculated)")
            st.metric("Statistical Parity Difference (SPD)",
                      f"{st.session_state.baseline_spd:.4f}")
            st.metric("Equal Opportunity Difference (EOD)",
                      f"{st.session_state.baseline_eod:.4f}")

            fig_fairness = plot_fairness_metrics(
                [st.session_state.baseline_spd], [st.session_state.baseline_eod], [
                    'Baseline'], "Baseline Model Fairness Metrics"
            )
            st.pyplot(fig_fairness)
            plt.close(fig_fairness)
            st.markdown("""
                *   **Interpretation of SPD:** If SPD is negative, it indicates that the privileged group has a higher rate of positive predictions than the unprivileged group. A positive SPD means the unprivileged group has a higher rate. The closer to 0, the better.
                *   **Interpretation of EOD:** If EOD is negative, it means the privileged group has a higher true positive rate (recall) than the unprivileged group. A positive EOD means the unprivileged group has a higher true positive rate. The closer to 0, the better.
                """)
        else:
            st.info(
                "Click the button above to calculate fairness metrics for the baseline model.")
