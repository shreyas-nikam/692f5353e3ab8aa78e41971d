
import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# Initialize session state variables
if 'df' not in st.session_state: st.session_state.df = None
if 'X_train_df' not in st.session_state: st.session_state.X_train_df = None
if 'X_test_df' not in st.session_state: st.session_state.X_test_df = None
if 'y_train' not in st.session_state: st.session_state.y_train = None
if 'y_test' not in st.session_state: st.session_state.y_test = None
if 'preprocessor' not in st.session_state: st.session_state.preprocessor = None
if 'all_feature_names' not in st.session_state: st.session_state.all_feature_names = None
if 'protected_attribute_names' not in st.session_state: st.session_state.protected_attribute_names = None
if 'aif360_privileged_groups' not in st.session_state: st.session_state.aif360_privileged_groups = None
if 'aif360_unprivileged_groups' not in st.session_state: st.session_state.aif360_unprivileged_groups = None
if 'baseline_model' not in st.session_state: st.session_state.baseline_model = None
if 'reweighed_model' not in st.session_state: st.session_state.reweighed_model = None
if 'baseline_accuracy' not in st.session_state: st.session_state.baseline_accuracy = None
if 'baseline_spd' not in st.session_state: st.session_state.baseline_spd = None
if 'baseline_eod' not in st.session_state: st.session_state.baseline_eod = None
if 'reweighed_accuracy' not in st.session_state: st.session_state.reweighed_accuracy = None
if 'reweighed_spd' not in st.session_state: st.session_state.reweighed_spd = None
if 'reweighed_eod' not in st.session_state: st.session_state.reweighed_eod = None
if 'adjusted_accuracy' not in st.session_state: st.session_state.adjusted_accuracy = None
if 'adjusted_spd' not in st.session_state: st.session_state.adjusted_spd = None
if 'adjusted_eod' not in st.session_state: st.session_state.adjusted_eod = None
if 'baseline_predictions' not in st.session_state: st.session_state.baseline_predictions = None
if 'reweighed_predictions' not in st.session_state: st.session_state.reweighed_predictions = None
if 'threshold_adjusted_predictions' not in st.session_state: st.session_state.threshold_adjusted_predictions = None
if 'shap_values_baseline' not in st.session_state: st.session_state.shap_values_baseline = None
if 'selected_instance_idx' not in st.session_state: st.session_state.selected_instance_idx = 0
if 'selected_dependence_feature' not in st.session_state: st.session_state.selected_dependence_feature = None
if 'selected_interaction_feature' not in st.session_state: st.session_state.selected_interaction_feature = None

st.set_page_config(page_title="QuLab", layout="wide")
st.sidebar.image("https://www.quantuniversity.com/assets/img/logo5.jpg")
st.sidebar.divider()
st.title("QuLab")
st.divider()
# Your code starts here
st.markdown("""
In this lab, we develop an **AI Credit Decision Explainer**, an interactive Streamlit application to provide insights into an AI-driven credit loan application scenario.
It allows users to train a classification model, understand its decisions through local and global explainability, identify and quantify biases, and evaluate bias mitigation techniques.

### Learning Goals
Upon completing the application experience, users will be able to:
*   Understand how local explainability tools (LIME) reveal factors influencing individual credit decisions.
*   Interpret global feature importance (SHAP) to grasp the model's overall decision-making patterns.
*   Identify potential biases in credit decisions across different demographic groups using fairness metrics (Statistical Parity Difference and Equal Opportunity Difference).
*   Evaluate the impact of simple mitigation techniques like reweighting and classification threshold adjustments on fairness and model performance.

### Target Audience
The application is designed for:
*   **Risk Managers**: To gain insights into how AI models make credit decisions, assess risks, and understand the factors driving approvals/denials.
*   **Executive Stakeholders**: To understand the trustworthiness, fairness, and transparency of AI systems in critical financial applications, enabling informed governance.
*   **Financial Data Engineers**: To understand the practical application of explainability and fairness tools, and how to implement basic bias detection and mitigation strategies.
""")

page = st.sidebar.selectbox(
    label="Navigation",
    options=[
        "Overview",
        "Data Preparation",
        "Baseline Model",
        "Explainability",
        "Fairness Analysis",
        "Bias Mitigation",
        "Comparative Analysis"
    ]
)

if page == "Overview":
    from application_pages.page_1_overview import main
    main()
elif page == "Data Preparation":
    from application_pages.page_2_data_preparation import main
    main()
elif page == "Baseline Model":
    from application_pages.page_3_baseline_model import main
    main()
elif page == "Explainability":
    from application_pages.page_4_explainability import main
    main()
elif page == "Fairness Analysis":
    from application_pages.page_5_fairness_analysis import main
    main()
elif page == "Bias Mitigation":
    from application_pages.page_6_bias_mitigation import main
    main()
elif page == "Comparative Analysis":
    from application_pages.page_7_comparative_analysis import main
    main()
# Your code ends here


# License
st.caption('''
---
## QuantUniversity License

© QuantUniversity 2025  
This notebook was created for **educational purposes only** and is **not intended for commercial use**.  

- You **may not copy, share, or redistribute** this notebook **without explicit permission** from QuantUniversity.  
- You **may not delete or modify this license cell** without authorization.  
- This notebook was generated using **QuCreate**, an AI-powered assistant.  
- Content generated by AI may contain **hallucinated or incorrect information**. Please **verify before using**.  

All rights reserved. For permissions or commercial licensing, contact: [info@qusandbox.com](mailto:info@qusandbox.com)
''')
