import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import shap
import lime
import lime.lime_tabular
import matplotlib.cm as cm

# --- Helper Functions ---


def plot_lime_explanation(explanation, title="LIME Explanation"):
    """Plots a LIME explanation."""
    fig = explanation.as_pyplot_figure()
    fig.suptitle(title)
    plt.tight_layout()
    return fig


def plot_shap_summary(shap_values, feature_names, title="SHAP Global Feature Importance"):
    """Plots a SHAP summary plot."""
    plt.figure(figsize=(10, 7))
    # For LinearExplainer, shap_values is a single array (for one class) or list of arrays (for multi-class)
    # For binary classification with LogisticRegression, shap_values is an array of shape (n_samples, n_features).
    # If it's a list, take the values for the positive class.
    if isinstance(shap_values, list):
        # Take SHAP values for the positive class (1)
        shap_values_to_plot = shap_values[1]
    else:
        shap_values_to_plot = shap_values

    shap.summary_plot(shap_values_to_plot, features=pd.DataFrame(
        shap_values_to_plot, columns=feature_names), feature_names=feature_names, show=False)
    plt.title(title)
    plt.tight_layout()
    return plt.gcf()


def plot_shap_dependence(shap_values, feature_names, feature, interaction_feature=None, title="SHAP Dependence Plot"):
    """Plots a SHAP dependence plot."""
    plt.figure(figsize=(10, 7))

    if isinstance(shap_values, list):
        # Take SHAP values for the positive class (1)
        shap_values_to_plot = shap_values[1]
    else:
        shap_values_to_plot = shap_values

    shap.dependence_plot(ind=feature,
                         shap_values=shap_values_to_plot,
                         features=pd.DataFrame(
                             shap_values_to_plot, columns=feature_names),
                         feature_names=feature_names,
                         interaction_index=interaction_feature,
                         show=False)
    plt.title(f"{title} for {feature}")
    plt.tight_layout()
    return plt.gcf()


def main():
    st.header("Explainability (LIME & SHAP)")

    st.markdown(r"""
    Understanding *why* an AI model makes a particular decision is crucial for trust and transparency, especially in sensitive domains like credit lending.
    This section explores two powerful explainability techniques: **LIME (Local Interpretable Model-agnostic Explanations)** for individual predictions and **SHAP (SHapley Additive exPlanations)** for global model behavior.

    ---

    ### 📖 Local Explainability: LIME

    LIME focuses on explaining individual predictions of any black-box classifier. It works by locally approximating the model around the prediction with an interpretable model (like a linear model).

    **How LIME Works:**
    1.  **Perturb the instance:** Generate perturbed samples around the instance we want to explain.
    2.  **Get predictions:** Use the black-box model to predict the outcome for these perturbed samples.
    3.  **Learn a local model:** Train a simple, interpretable model (e.g., linear regression) on the perturbed samples, weighted by their proximity to the original instance.
    4.  **Explain:** The coefficients of this local model provide an explanation for the individual prediction.

    #### Mathematical Concept (Simplified):
    For a given prediction $f(x)$, LIME seeks to find an interpretable model $g$ that locally approximates $f$ around $x$.
    $$ \xi(x) = \underset{g \in G}{\arg\min} L(f, g, \pi_x) + \Omega(g) $$
    Where:
    *   $g$ is an interpretable model (e.g., a linear model).
    *   $G$ is the class of interpretable models.
    *   $L(f, g, \pi_x)$ is a fidelity function that measures how well $g$ approximates $f$ in the vicinity defined by $\pi_x$.
    *   $\Omega(g)$ is a complexity measure for the interpretable model $g$.

    """)

    # Ensure necessary data and model are in session state
    if "X_test_df" not in st.session_state or "y_test" not in st.session_state or \
       "baseline_model" not in st.session_state or "all_feature_names" not in st.session_state:
        st.warning("Please train the 'Baseline Model' first.")
        return

    X_test_df = st.session_state.X_test_df
    all_feature_names = st.session_state.all_feature_names
    baseline_model = st.session_state.baseline_model

    st.subheader("LIME: Explaining Individual Predictions")

    # Select instance for LIME
    max_idx = len(X_test_df) - 1
    selected_instance_idx = st.slider(
        "Select Test Instance Index for LIME Explanation", min_value=0, max_value=max_idx, value=0, key="lime_idx")
    st.session_state.selected_instance_idx = selected_instance_idx

    st.markdown(f"**Selected instance details (first 5 features):**")
    st.dataframe(X_test_df.iloc[[selected_instance_idx]].head())
    st.write(
        f"**True Label:** {st.session_state.y_test.iloc[selected_instance_idx]} (0=Bad Credit, 1=Good Credit)")
    st.write(
        f"**Predicted Label:** {baseline_model.predict(X_test_df.iloc[[selected_instance_idx]])[0]} (0=Bad Credit, 1=Good Credit)")
    st.write(
        f"**Predicted Probability (Good Credit):** {baseline_model.predict_proba(X_test_df.iloc[[selected_instance_idx]])[0, 1]:.4f}")

    if st.button("Generate LIME Explanation"):
        with st.spinner("Generating LIME explanation..."):
            try:
                explainer = lime.lime_tabular.LimeTabularExplainer(
                    training_data=st.session_state.X_train_df.values,
                    feature_names=all_feature_names,
                    class_names=['Bad Credit (0)', 'Good Credit (1)'],
                    mode='classification'
                )

                # Get explanation for the selected instance
                explanation = explainer.explain_instance(
                    data_row=X_test_df.iloc[selected_instance_idx].values,
                    predict_fn=baseline_model.predict_proba,
                    num_features=10  # Explain top 10 features
                )

                st.session_state.lime_explanation = explanation

                st.subheader("LIME Explanation for Selected Instance")
                st.info("The bars represent the contribution of each feature to the model's prediction for this specific instance. Positive values (green) indicate features pushing towards 'Good Credit', while negative values (red) push towards 'Bad Credit'.")

                fig_lime = plot_lime_explanation(
                    explanation, f"LIME Explanation for Instance {selected_instance_idx}")
                st.pyplot(fig_lime)
                plt.close(fig_lime)

            except Exception as e:
                st.error(f"Error generating LIME explanation: {e}")

    if "lime_explanation" in st.session_state:
        st.subheader("Current LIME Explanation for Instance " +
                     str(st.session_state.selected_instance_idx))
        st.info("The bars represent the contribution of each feature to the model's prediction for this specific instance. Positive values (green) indicate features pushing towards 'Good Credit', while negative values (red) push towards 'Bad Credit'.")
        fig_lime = plot_lime_explanation(
            st.session_state.lime_explanation, f"LIME Explanation for Instance {st.session_state.selected_instance_idx}")
        st.pyplot(fig_lime)
        plt.close(fig_lime)

    st.markdown(r"""
    ---

    ### 📈 Global Explainability: SHAP (SHapley Additive exPlanations)

    SHAP values unify several explainability methods by assigning to each feature an importance value for a particular prediction. It's based on game theory's Shapley values, which distribute the total gain among cooperating players.

    **How SHAP Works:**
    SHAP values quantify how much each feature contributes to the difference between the actual prediction and the average prediction.

    #### Mathematical Concept (Simplified):
    The SHAP value $\phi_i$ for a feature $i$ is calculated by averaging the marginal contribution of that feature across all possible coalitions (subsets of features).
    $$ \phi_i(f, x) = \sum_{S \subseteq F \setminus \{i\}} \frac{|S|!(|F| - |S| - 1)!}{|F|!} (f_x(S \cup \{i\}) - f_x(S)) $$
    Where:
    *   $F$ is the set of all features.
    *   $S$ is a subset of features.
    *   $f_x(S)$ is the prediction using only features in set $S$.

    SHAP provides two main types of plots for global understanding:
    *   **Summary Plot:** Shows the overall impact of features on the model output.
    *   **Dependence Plot:** Visualizes the effect of a single feature on the prediction and how it interacts with another feature.
    """)

    st.subheader("SHAP: Global Model Understanding")

    # SHAP explainer setup (only run once)
    if "shap_explainer_baseline" not in st.session_state:
        with st.spinner("Initializing SHAP Explainer (this may take a moment)..."):
            try:
                # Create a masker for the background data
                masker = shap.maskers.Independent(
                    data=st.session_state.X_train_df)
                st.session_state.shap_explainer_baseline = shap.LinearExplainer(
                    model=baseline_model,
                    masker=masker
                )
                st.success("SHAP Explainer initialized.")
            except Exception as e:
                st.error(f"Error initializing SHAP explainer: {e}")

    if "shap_values_baseline" not in st.session_state and "shap_explainer_baseline" in st.session_state:
        with st.spinner("Calculating SHAP values for the test set (this may take a moment)..."):
            try:
                st.session_state.shap_values_baseline = st.session_state.shap_explainer_baseline.shap_values(
                    st.session_state.X_test_df)
                st.success("SHAP values calculated.")
            except Exception as e:
                st.error(f"Error calculating SHAP values: {e}")

    if "shap_values_baseline" in st.session_state:
        st.markdown("#### SHAP Summary Plot: Global Feature Importance")
        st.info("""
        The SHAP summary plot shows the overall impact of features on the model's output.
        *   **X-axis:** SHAP value (impact on model output).
        *   **Y-axis:** Features ordered by importance.
        *   **Color:** Represents the feature value (e.g., red for high, blue for low).
        A high SHAP value (to the right) means the feature is pushing the prediction towards 'Good Credit'.
        """)
        fig_shap_summary = plot_shap_summary(
            st.session_state.shap_values_baseline, all_feature_names, "SHAP Summary Plot for Baseline Model")
        st.pyplot(fig_shap_summary)
        plt.close(fig_shap_summary)

        st.markdown("---")
        st.markdown(
            "#### SHAP Dependence Plot: Feature Effect and Interactions")

        st.info("""
        The SHAP dependence plot illustrates how the value of a single feature affects the prediction.
        Each point is an instance from the test dataset.
        *   **X-axis:** Value of the selected feature.
        *   **Y-axis:** SHAP value for that feature (impact on prediction).
        *   **Color (if interaction feature selected):** Represents the value of the interaction feature, showing how its value influences the effect of the primary feature.
        """)

        # Select feature for dependence plot
        selected_dependence_feature = st.selectbox(
            "Select Primary Feature for Dependence Plot",
            options=all_feature_names,
            index=all_feature_names.index(
                "duration_in_month") if "duration_in_month" in all_feature_names else 0,
            key="shap_dep_feature"
        )
        st.session_state.selected_dependence_feature = selected_dependence_feature

        # Select interaction feature
        interaction_options = ["None"] + all_feature_names
        selected_interaction_feature = st.selectbox(
            "Select Interaction Feature (Optional)",
            options=interaction_options,
            index=interaction_options.index("age_years") if "age_years" in interaction_options else (
                0 if len(interaction_options) > 1 else 0),
            key="shap_inter_feature"
        )
        st.session_state.selected_interaction_feature = selected_interaction_feature

        if st.button("Generate SHAP Dependence Plot"):
            with st.spinner(f"Generating SHAP Dependence Plot for {selected_dependence_feature}..."):
                try:
                    interaction_feat_for_plot = None if selected_interaction_feature == "None" else selected_interaction_feature
                    fig_shap_dependence = plot_shap_dependence(
                        st.session_state.shap_values_baseline,
                        all_feature_names,
                        selected_dependence_feature,
                        interaction_feature=interaction_feat_for_plot,
                        title="SHAP Dependence Plot for Baseline Model"
                    )
                    st.pyplot(fig_shap_dependence)
                    plt.close(fig_shap_dependence)
                except Exception as e:
                    st.error(f"Error generating SHAP dependence plot: {e}")
    else:
        st.info("SHAP values are not yet calculated. This happens automatically after explainer initialization. If you just navigated here, wait for it to compute or refresh.")
