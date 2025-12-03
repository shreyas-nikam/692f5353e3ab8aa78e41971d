import streamlit as st
import matplotlib.pyplot as plt
from utils import apply_reweighing_and_retrain, evaluate_model, calculate_fairness_metrics, apply_threshold_adjustment_and_evaluate

def main():
    st.header("Section 12: Bias Mitigation Technique: Reweighting")
    st.markdown(r"""
    **Reweighting** is a pre-processing bias mitigation technique that adjusts the weights of individual training samples to reduce disparities in outcomes for protected groups. It aims to create a more balanced dataset effectively by giving more importance to underrepresented or unfairly treated groups.

    The principle involves adjusting sample weights $w_i$ in the loss function during training, typically to equalize the representation of protected groups. For a sample $(x_i, y_i)$, the reweighted loss term might be $w_i \cdot Loss(y_i, \hat{y}_i)$. By assigning higher weights to instances from unprivileged groups or those with unfavorable outcomes, the model learns to pay more attention to these samples, thus reducing bias.
    """)
    if st.button("Apply Reweighing & Retrain Model"):
        if st.session_state.get('X_train_df') is not None:
            with st.spinner("Applying Reweighing and retraining model..."):
                st.session_state.reweighed_model = apply_reweighing_and_retrain(
                    st.session_state.X_train_df,
                    st.session_state.y_train,
                    st.session_state.protected_attribute_names,
                    st.session_state.aif360_privileged_groups,
                    st.session_state.aif360_unprivileged_groups
                )
                st.success("Model retrained after applying Reweighing mitigation.")
                st.header("Section 13: Evaluating Reweighted Model Performance and Fairness")
                st.markdown("After applying the reweighting technique, we need to evaluate its impact on both model performance (accuracy, precision, recall, F1-score) and fairness metrics (SPD, EOD). We will compare these results against the baseline model to assess the effectiveness and any potential trade-offs.")
                st.session_state.reweighed_accuracy, _, _, _, _, st.session_state.reweighed_predictions = \
                    evaluate_model(st.session_state.reweighed_model, st.session_state.X_test_df, st.session_state.y_test, "Reweighed Model")
                st.session_state.reweighed_spd, st.session_state.reweighed_eod, reweighed_fairness_fig = calculate_fairness_metrics(
                    st.session_state.reweighed_model,
                    st.session_state.X_test_df,
                    st.session_state.y_test,
                    st.session_state.protected_attribute_names,
                    st.session_state.aif360_privileged_groups,
                    st.session_state.aif360_unprivileged_groups,
                    model_name="Reweighed Model"
                )
                st.pyplot(reweighed_fairness_fig)
                plt.close(reweighed_fairness_fig)
                st.markdown("""
                The reweighted model shows its accuracy and updated fairness metrics. We can observe whether reweighting successfully reduced the bias as measured by SPD and EOD. There might be a slight change in overall accuracy as fairness-accuracy trade-offs are common in bias mitigation.
                """)
        else:
            st.warning("Please load and preprocess data first.")

    st.header("Section 14: Bias Mitigation Technique: Threshold Adjustment")
    st.markdown(r"""
    **Threshold Adjustment** is a post-processing bias mitigation technique. Instead of changing the model or the data, it modifies the classification threshold ($T$) after the model has made its predictions. This allows for achieving fairness targets by adjusting the threshold for different demographic groups.

    For example, for a group $a_0$, the prediction $\hat{Y}=1$ if $P(\hat{Y}=1|A=a_0) > T_{a_0}$. By optimizing group-specific thresholds, we can equalize metrics like True Positive Rate (to achieve Equal Opportunity) or Positive Prediction Rate (to achieve Statistical Parity) across groups.
    """)
    if st.button("Apply Threshold Adjustment & Re-evaluate Model"):
        if st.session_state.get('baseline_model') is not None:
            with st.spinner("Applying threshold adjustment and re-evaluating..."):
                st.session_state.adjusted_accuracy, st.session_state.adjusted_spd, st.session_state.adjusted_eod, st.session_state.threshold_adjusted_predictions, adjusted_fairness_fig = \
                    apply_threshold_adjustment_and_evaluate(
                        st.session_state.baseline_model,
                        st.session_state.X_train_df,
                        st.session_state.y_train,
                        st.session_state.X_test_df,
                        st.session_state.y_test,
                        st.session_state.protected_attribute_names,
                        st.session_state.aif360_privileged_groups,
                        st.session_state.aif360_unprivileged_groups
                    )
                st.pyplot(adjusted_fairness_fig)
                plt.close(adjusted_fairness_fig)
                st.markdown("""
                By applying Calibrated Equalized Odds Postprocessing, the classification thresholds were adjusted for different groups. This technique directly intervenes on the model's output to balance fairness metrics, potentially leading to a more equitable distribution of loan approvals while aiming to maintain performance.
                """)
        else:
            st.warning("Please train the baseline model first.")