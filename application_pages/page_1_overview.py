
import streamlit as st

def main():
    st.markdown("""
    ## 1. Notebook Overview
    ### Learning Goals
    Upon completing this notebook, users will be able to:
    *   Understand how local explainability tools (LIME) reveal factors influencing individual credit decisions.
    *   Interpret global feature importance (SHAP) to grasp the model's overall decision-making patterns.
    *   Identify potential biases in credit decisions across different demographic groups using fairness metrics (Statistical Parity Difference and Equal Opportunity Difference).
    *   Evaluate the impact of simple mitigation techniques like reweighting and classification threshold adjustments on fairness and model performance.

    ### Target Audience
    This notebook is designed for:
    *   **Risk Managers**: To gain insights into how AI models make credit decisions, assess risks, and understand the factors driving approvals/denials.
    *   **Executive Stakeholders**: To understand the trustworthiness, fairness, and transparency of AI systems in critical financial applications, enabling informed governance.
    *   **Financial Data Engineers**: To understand the practical application of explainability and fairness tools, and how to implement basic bias detection and mitigation strategies.

    ## Section 1: Introduction to AI Credit Decision Explainer
    This section introduces the purpose of the notebook: to simulate an AI-driven credit loan application scenario, understand its decisions, identify potential biases, and explore mitigation techniques. It will set the stage for Risk Managers, Executive Stakeholders, and Financial Data Engineers to comprehend model behavior, trust, and fairness.

    The concept of Explainable AI (XAI) is critical for trust and accountability. We will explore:
    *   **Local Interpretability** (LIME): Explaining individual predictions.
    *   **Global Interpretability** (SHAP): Understanding overall model behavior.
    *   **Algorithmic Fairness**: Detecting and mitigating biases.
    """)
