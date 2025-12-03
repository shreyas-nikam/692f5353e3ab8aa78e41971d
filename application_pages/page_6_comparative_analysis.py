import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns


def plot_comparative_metrics(metrics_values, labels, metric_name, title):
    """Plots comparative bar charts for given metrics."""

    df = pd.DataFrame({
        'Model': labels,
        metric_name: metrics_values
    })

    fig, ax = plt.subplots(figsize=(8, 6))
    sns.barplot(x='Model', y=metric_name, data=df, palette='viridis', ax=ax)

    if metric_name in ['SPD', 'EOD']:
        ax.axhline(0, color='grey', linestyle='--', linewidth=0.8)

    ax.set_title(title)
    ax.set_ylabel(metric_name)
    ax.set_xlabel("Model Type")
    plt.xticks(rotation=45, ha='right')
    plt.tight_layout()
    return fig


def main():
    st.header("Comparative Analysis of Models")

    st.markdown("""
    This section provides a comparative view of the **Baseline Model**, the **Reweighed Model**, and the **Threshold-Adjusted Model**.
    We will compare their performance (Accuracy) and fairness (Statistical Parity Difference and Equal Opportunity Difference) to understand the trade-offs involved in bias mitigation.
    """)

    # Ensure all necessary metrics are in session state
    metrics_to_check = [
        "baseline_accuracy", "baseline_spd", "baseline_eod",
        "reweighed_accuracy", "reweighed_spd", "reweighed_eod",
        "adjusted_accuracy", "adjusted_spd", "adjusted_eod"
    ]

    missing_metrics = [
        m for m in metrics_to_check if m not in st.session_state or st.session_state[m] is None]

    if missing_metrics:
        st.warning(
            f"Please ensure all models are trained and evaluated. Missing metrics: {', '.join(missing_metrics)}")
        st.info("Navigate to 'Baseline Model Training', 'Fairness Analysis', and 'Bias Mitigation' to complete all steps.")
        return

    st.subheader("Comparison of Performance and Fairness Metrics")

    # Collect all metrics
    labels = ["Baseline", "Reweighed", "Threshold Adjusted"]

    accuracies = [
        st.session_state.baseline_accuracy,
        st.session_state.reweighed_accuracy,
        st.session_state.adjusted_accuracy
    ]

    spds = [
        st.session_state.baseline_spd,
        st.session_state.reweighed_spd,
        st.session_state.adjusted_spd
    ]

    eods = [
        st.session_state.baseline_eod,
        st.session_state.reweighed_eod,
        st.session_state.adjusted_eod
    ]

    # Display in a table first
    comparison_df = pd.DataFrame({
        "Model": labels,
        "Accuracy": [f"{acc:.4f}" for acc in accuracies],
        "SPD": [f"{s:.4f}" for s in spds],
        "EOD": [f"{e:.4f}" for e in eods]
    })
    st.dataframe(comparison_df)

    st.markdown("---")
    st.subheader("Visual Comparison")

    # Plot Accuracy
    fig_accuracy = plot_comparative_metrics(
        accuracies, labels, 'Accuracy', 'Model Accuracy Comparison')
    st.pyplot(fig_accuracy)
    plt.close(fig_accuracy)
    st.info("Higher accuracy is generally better, but we need to consider fairness trade-offs.")

    # Plot SPD
    fig_spd = plot_comparative_metrics(
        spds, labels, 'SPD', 'Statistical Parity Difference Comparison')
    st.pyplot(fig_spd)
    plt.close(fig_spd)
    st.info("SPD closer to 0 indicates better statistical parity (equal positive prediction rates across groups).")

    # Plot EOD
    fig_eod = plot_comparative_metrics(
        eods, labels, 'EOD', 'Equal Opportunity Difference Comparison')
    st.pyplot(fig_eod)
    plt.close(fig_eod)
    st.info("EOD closer to 0 indicates better equal opportunity (equal true positive rates across groups).")

    st.markdown("""
    #### Interpretation of Results:
    *   Observe how bias mitigation techniques impact both model performance (accuracy) and fairness metrics (SPD, EOD).
    *   Often, there's a **trade-off** between maximizing accuracy and achieving perfect fairness. Mitigating bias might lead to a slight decrease in overall accuracy, but it results in a more equitable model.
    *   **Reweighing** (a pre-processing method) adjusts the training data.
    *   **Threshold Adjustment** (a post-processing method) adjusts predictions without retraining the model.

    By comparing these metrics, you can evaluate which mitigation strategy best aligns with the desired balance between performance and fairness objectives for your specific application.
    """)
