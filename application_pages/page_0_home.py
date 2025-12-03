import streamlit as st

def main():
    st.header("Welcome to the AI Credit Decision Explainer")
    st.markdown("""
    This interactive Streamlit application provides a comprehensive toolkit for understanding and evaluating AI models used in credit loan applications.
    It focuses on bringing transparency, explainability, and fairness to the forefront of credit decision-making.

    ---

    ### 🎓 Learning Goals

    Upon completing your interaction with this application, you will be able to:

    *   **Understand Local Explainability (LIME):** Discover how LIME reveals the specific factors influencing individual credit decisions (e.g., why a particular applicant was approved or denied).
    *   **Interpret Global Feature Importance (SHAP):** Grasp the model's overall decision-making patterns by identifying the most impactful features across the entire dataset.
    *   **Identify Potential Biases:** Calculate and visualize fairness metrics like Statistical Parity Difference (SPD) and Equal Opportunity Difference (EOD) to detect disparities in credit outcomes across different demographic groups.
    *   **Evaluate Bias Mitigation Techniques:** Assess the effectiveness of simple mitigation strategies, such as reweighting and classification threshold adjustments, in improving fairness while maintaining model performance.

    ---

    ### 🎯 Target Audience

    This application is designed for a diverse group of professionals interested in responsible AI in finance:

    *   **Risk Managers**: Gain deep insights into AI model behavior, assess associated risks, and understand the factors driving credit approvals and denials. This tool helps in scrutinizing model decisions for regulatory compliance and internal risk assessment.
    *   **Executive Stakeholders**: Understand the trustworthiness, fairness, and transparency of AI systems deployed in critical financial applications. It facilitates informed governance and strategic decision-making regarding AI adoption.
    *   **Financial Data Engineers & Scientists**: Learn the practical application of explainability and fairness tools. Explore how to implement basic bias detection and mitigation strategies within a real-world context, enhancing skills in building ethical AI solutions.

    ---

    ### 🚀 Get Started

    Navigate through the sections using the sidebar to explore data, train models, understand explanations, analyze fairness, and experiment with bias mitigation strategies.
    """)