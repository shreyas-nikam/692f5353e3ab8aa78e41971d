import streamlit as st
import matplotlib.pyplot as plt
from utils import plot_comparative_metrics

def main():
    st.header("Section 15: Comparative Analysis of Mitigation Techniques")
    st.markdown("""
    This section provides a comparative overview of the baseline model's performance and fairness, versus the models after applying reweighting and threshold adjustment. This allows for a direct assessment of the trade-offs between model accuracy and fairness improvements achieved by each mitigation strategy.
    """)
    if all(k in st.session_state for k in ['baseline_accuracy', 'baseline_spd', 'baseline_eod',
                                          'reweighed_accuracy', 'reweighed_spd', 'reweighed_eod',
                                          'adjusted_accuracy', 'adjusted_spd', 'adjusted_eod']):
        if st.button("Plot Comparative Metrics"):
            with st.spinner("Generating comparative plots..."):
                comp_fig = plot_comparative_metrics(
                    st.session_state.baseline_accuracy, st.session_state.baseline_spd, st.session_state.baseline_eod,
                    st.session_state.reweighed_accuracy, st.session_state.reweighed_spd, st.session_state.reweighed_eod,
                    st.session_state.adjusted_accuracy, st.session_state.adjusted_spd, st.session_state.adjusted_eod
                )
                st.pyplot(comp_fig)
                plt.close(comp_fig)
                st.markdown("""
                The comparative plots clearly illustrate the impact of each mitigation technique. We can observe how reweighting and threshold adjustment affected accuracy, SPD, and EOD relative to the baseline. This allows Risk Managers and Executive Stakeholders to make informed decisions about which mitigation strategy best aligns with their organizational goals and risk appetite, considering the trade-offs between predictive performance and fairness. For example, one technique might achieve better fairness but at a greater cost to overall accuracy.
                """)
    else:
        st.warning("Please train the baseline model and apply mitigation techniques to see the comparative analysis.")

    st.header("Section 16: Summary and Key Takeaways")
    st.markdown("""
    This notebook demonstrated the importance of Explainable AI (XAI) and fairness in critical applications like credit decisions. We covered:
    1.  **Local Explanations (LIME)**: Provided individual-level insights into why a specific loan applicant received a particular decision, detailing feature contributions. This directly addresses the 'right to explanation'.
    2.  **Global Explanations (SHAP)**: Revealed the overall drivers of the model's decisions, showing which features are generally most influential for loan approvals.
    3.  **Fairness Assessment**: Quantified bias using Statistical Parity Difference (SPD) and Equal Opportunity Difference (EOD) for a protected attribute (Gender).
    4.  **Bias Mitigation**: Explored and evaluated two techniques:
        *   **Reweighting (Pre-processing)**: Adjusted training sample weights to balance group representation.
        *   **Threshold Adjustment (Post-processing)**: Modified classification thresholds on model outputs to equalize fairness metrics.

    **Key Takeaways**:
    *   XAI tools (LIME, SHAP) are indispensable for building trust and transparency in AI models, allowing stakeholders to understand both individual predictions and overall model behavior.
    *   Fairness metrics like SPD and EOD are essential for detecting and quantifying algorithmic bias, particularly for protected demographic groups.
    *   Bias mitigation techniques, whether applied during preprocessing (reweighting) or post-processing (threshold adjustment), can effectively reduce disparities but often involve trade-offs with model performance.
    *   Understanding these trade-offs is crucial for responsible AI deployment, enabling organizations to balance predictive accuracy with ethical considerations and regulatory compliance.
    """)