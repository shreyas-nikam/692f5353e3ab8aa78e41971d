import streamlit as st
import matplotlib.pyplot as plt
from utils import calculate_fairness_metrics

def main():
    st.header("Section 10: Introduction to Fairness Metrics")
    st.markdown(r"""
    Algorithmic fairness is crucial in high-stakes decisions like credit approval. We will use two common fairness metrics:
    *   **Statistical Parity Difference (SPD)**: Measures the difference in the proportion of positive outcomes (loan approvals) between the unprivileged group ($a_0$) and the privileged group ($a_1$). A value of 0 indicates perfect statistical parity.
        $$ SPD = P(\hat{Y}=1|A=a_1) - P(\hat{Y}=1|A=a_0) $$
        where $P(\hat{Y}=1|A=a_i)$ is the probability of a positive outcome (loan approval) for group $a_i$. A positive SPD means the privileged group is more likely to be approved.

    *   **Equal Opportunity Difference (EOD)**: Measures the difference in the true positive rates (recall for the positive class) between the unprivileged group ($a_0$) and the privileged group ($a_1$), specifically among those who are truly qualified (actual $Y=1$). A value of 0 indicates perfect equal opportunity.
        $$ EOD = P(\hat{Y}=1|A=a_1, Y=1) - P(\hat{Y}=1|A=a_0, Y=1) $$
        where $P(\hat{Y}=1|A=a_i, Y=1)$ is the true positive rate for group $a_i$. A positive EOD means the privileged group is more likely to be correctly approved when they are truly creditworthy.
    """)

    st.header("Section 11: Calculating Baseline Fairness Metrics")
    st.markdown("We will calculate SPD and EOD for our baseline Logistic Regression model using `aif360`. We define 'Gender_1' (male) as the privileged group and 'Gender_0' (female) as the unprivileged group. The favorable label for loan approval is 1.")

    if st.button("Calculate Baseline Fairness Metrics"):
        if st.session_state.get('baseline_model') is not None and st.session_state.get('X_test_df') is not None:
            with st.spinner("Calculating baseline fairness metrics..."):
                st.session_state.baseline_spd, st.session_state.baseline_eod, baseline_fairness_fig = calculate_fairness_metrics(
                    st.session_state.baseline_model,
                    st.session_state.X_test_df,
                    st.session_state.y_test,
                    st.session_state.protected_attribute_names,
                    st.session_state.aif360_privileged_groups,
                    st.session_state.aif360_unprivileged_groups,
                    model_name="Baseline Model"
                )
                st.pyplot(baseline_fairness_fig)
                plt.close(baseline_fairness_fig)
                st.markdown("""
                The baseline model's fairness metrics show the extent of disparate impact and unequal opportunity between the male (privileged) and female (unprivileged) groups regarding loan approvals. A non-zero value suggests bias, with positive values indicating bias favoring the privileged group. The bar chart visually represents these disparities.
                """)
        else:
            st.warning("Please train the baseline model first.")
