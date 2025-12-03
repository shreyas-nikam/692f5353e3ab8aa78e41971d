import streamlit as st
import matplotlib.pyplot as plt
from utils import generate_lime_explanation, generate_shap_summary, generate_shap_dependence_plot

def main():
    st.header("Section 7: Local Explainability with LIME")
    st.markdown(r"""
    **LIME (Local Interpretable Model-agnostic Explanations)** is a technique to explain individual predictions of any black-box model. It works by training a simple, interpretable model (like a linear model or decision tree) locally around the prediction point. The core idea is to approximate the complex model locally with an interpretable one.

    The weighted loss function minimized by LIME is:
    $$ L(f, g, \pi_x) = \sum_{z \in Z} \pi_x(z) (f(z) - g(z))^2 $$
    where $f$ is the black-box model, $g$ is the interpretable model, $Z$ is the set of perturbed samples, and $\pi_x(z)$ is the proximity measure of $z$ to the instance $x$ being explained.
    """)
    if st.session_state.get('baseline_model') is not None:
        max_idx = len(st.session_state.X_test_df) - 1
        if max_idx >= 0:
            st.session_state.selected_instance_idx = st.slider("Select Test Instance Index for LIME", 0, max_idx, 0)
            if st.button("Generate LIME Explanation"):
                with st.spinner("Generating LIME explanation..."):
                    lime_fig = generate_lime_explanation(
                        st.session_state.baseline_model,
                        st.session_state.X_train_df,
                        st.session_state.X_test_df,
                        st.session_state.all_feature_names,
                        st.session_state.selected_instance_idx
                    )
                    st.pyplot(lime_fig)
                    plt.close(lime_fig)
                    st.markdown("""
                    A LIME explanation has been generated for a selected applicant from the test set. The plot shows the features that most influenced the model's prediction for this specific individual. Positive contributions indicate features pushing towards 'Approved' (Class 1), while negative contributions push towards 'Denied' (Class 0). For instance, if 'LoanAmount' is high and contributes negatively, it suggests that for this applicant, a high loan amount was a factor against approval.
                    """)
        else:
            st.warning("Test dataset is empty. Cannot generate LIME explanation.")
    else:
        st.warning("Please train the baseline model first.")

    st.header("Section 8: Global Explainability with SHAP")
    st.markdown(r"""
    **SHAP (SHapley Additive exPlanations)** provides a unified framework to explain predictions by assigning each feature an importance value for a particular prediction. It is based on Shapley values from cooperative game theory. Each feature's contribution is the average marginal contribution across all possible coalitions of features.

    The SHAP value $\phi_j$ for a feature $j$ is calculated as:
    $$ \phi_j = \sum_{S \subseteq N \setminus \{j\}} \frac{|S|!(|N| - |S| - 1)!}{|N|!} [f_x(S \cup \{j\}) - f_x(S)] $$
    where $N$ is the set of all features, $S$ is a subset of features, and $f_x(S)$ is the model's prediction with features in $S$ present.
    """)
    if st.session_state.get('baseline_model') is not None:
        if st.button("Generate SHAP Summary Plot"):
            with st.spinner("Generating SHAP summary plot..."):
                shap_fig, st.session_state.shap_values_baseline = generate_shap_summary(
                    st.session_state.baseline_model,
                    st.session_state.X_train_df,
                    st.session_state.X_test_df,
                    st.session_state.all_feature_names
                )
                st.pyplot(shap_fig)
                plt.close(shap_fig)
                st.markdown("""
                The SHAP summary plot provides a global view of feature importance. Each point represents a Shapley value for a feature and an instance. The color indicates the feature value (red for high, blue for low), and the position on the x-axis indicates the impact on the model's output (higher SHAP value means higher prediction of 'Approved'). Features are ordered by their overall impact. This helps Risk Managers understand the primary drivers of credit approval across the entire portfolio.
                """)
    else:
        st.warning("Please train the baseline model first.")

    st.header("Section 9: SHAP Dependence Plots")
    st.markdown("""
    SHAP dependence plots show how the value of a single feature affects the prediction of the model, often revealing non-linear relationships or interactions with other features. Each point on the plot is a single prediction from the dataset.
    """)
    if st.session_state.get('shap_values_baseline') is not None and st.session_state.get('all_feature_names') is not None:
        st.session_state.selected_dependence_feature = st.selectbox(
            "Select Feature for Dependence Plot",
            st.session_state.all_feature_names,
            index=st.session_state.all_feature_names.index('CreditDuration') if 'CreditDuration' in st.session_state.all_feature_names else 0
        )
        interaction_options = ['None'] + st.session_state.all_feature_names
        st.session_state.selected_interaction_feature = st.selectbox(
            "Select Interaction Feature (Optional)",
            interaction_options,
            index=interaction_options.index('LoanAmount') if 'LoanAmount' in interaction_options else 0
        )
        interaction_feat = None if st.session_state.selected_interaction_feature == 'None' else st.session_state.selected_interaction_feature

        if st.button("Generate SHAP Dependence Plot"):
            with st.spinner("Generating SHAP dependence plot..."):
                dep_fig = generate_shap_dependence_plot(
                    st.session_state.shap_values_baseline,
                    st.session_state.X_test_df,
                    st.session_state.selected_dependence_feature,
                    interaction_feat
                )
                st.pyplot(dep_fig)
                plt.close(dep_fig)
                st.markdown("""
                The SHAP dependence plot illustrates how the value of the selected feature impacts the prediction for loan approval. We can observe patterns such as whether longer or shorter credit durations tend to increase or decrease the probability of approval. If an interaction feature was used, the color variation would indicate how the interaction with that feature further influences the prediction. This helps understand specific feature behaviors.
                """)
    else:
        st.warning("Please generate SHAP summary plot first to enable dependence plots."))