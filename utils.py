import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix, classification_report
import lime
import lime.lime_tabular
import shap
from aif360.datasets import StandardDataset
from aif360.metrics import BinaryLabelDatasetMetric, ClassificationMetric
from aif360.algorithms.preprocessing import Reweighing
from aif360.algorithms.postprocessing import CalibratedEqOddsPostprocessing
import warnings

# Suppress warnings and set plotting style
warnings.filterwarnings("ignore")
sns.set_style("whitegrid")
plt.rcParams.update({"figure.max_open_warning": 0}) # Suppress matplotlib warning for too many figures

# Section 3: Dataset Loading and Initial Exploration
def load_and_explore_data(file_path):
    """
    Loads the German Credit Data and displays its head, info, and shape.
    """
    data = pd.read_csv(file_path)
    return data

# Section 4: Data Preprocessing for Modeling
def preprocess_data(data):
    """
    Preprocesses the German Credit Data: renames columns, maps target,
    creates age groups, defines protected attributes, and splits data.
    """
    df_processed = data.copy()
    df_processed = df_processed.rename(columns={
        'Duration_of_Credit_month': 'CreditDuration',
        'Credit_Amount': 'LoanAmount',
        'Installation_Rate_in_Percentage_of_Disposable_Income': 'InstallmentRate',
        'Present_Residence_since': 'YearsResidence',
        'Age_in_years': 'Age',
        'Number_of_Credits_at_this_Bank': 'NumCredits',
        'Number_of_people_being_liable_to_provide_maintenance_for': 'NumDependents',
        'Class': 'LoanApproved'
    })
    df_processed['LoanApproved'] = df_processed['LoanApproved'].map({1: 1, 2: 0})
    df_processed['Gender'] = df_processed['Personal_Status_and_Sex'].apply(
        lambda x: 1 if 'male' in x.lower() else 0
    )
    bins = [18, 25, 35, 45, 55, 65, 100]
    labels = ['18-24', '25-34', '35-44', '45-54', '55-64', '65+']
    df_processed['AgeGroup'] = pd.cut(df_processed['Age'], bins=bins, labels=labels, right=False)
    df_processed['AgeGroup'] = df_processed['AgeGroup'].astype(str)
    df_processed = df_processed.drop(columns=['Personal_Status_and_Sex'])
    X = df_processed.drop('LoanApproved', axis=1)
    y = df_processed['LoanApproved']
    categorical_features = X.select_dtypes(include='object').columns
    numerical_features = X.select_dtypes(include=['int64', 'float64']).columns
    numerical_transformer = StandardScaler()
    categorical_transformer = OneHotEncoder(handle_unknown='ignore')
    preprocessor = ColumnTransformer(
        transformers=[
            ('num', numerical_transformer, numerical_features),
            ('cat', categorical_transformer, categorical_features)
        ],
        remainder='passthrough'
    )
    return X, y, preprocessor, categorical_features, numerical_features, df_processed.columns.tolist()

# Section 5: Baseline Model Training (Logistic Regression)
def train_model(X_train, y_train, sample_weight=None):
    """
    Trains a Logistic Regression model.
    """
    model = LogisticRegression(solver='liblinear', random_state=42)
    model.fit(X_train, y_train, sample_weight=sample_weight)
    return model

# Section 6: Baseline Model Evaluation
def evaluate_model(model, X_test, y_test, name="Model"):
    """
    Evaluates a given model and prints classification metrics.
    Returns metrics and predictions.
    """
    y_pred = model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred)
    recall = recall_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)
    conf_matrix = confusion_matrix(y_test, y_pred)

    st.subheader(f"--- {name} Evaluation ---")
    st.metric(label="Accuracy", value=f"{accuracy:.4f}")
    st.metric(label="Precision", value=f"{precision:.4f}")
    st.metric(label="Recall", value=f"{recall:.4f}")
    st.metric(label="F1-Score", value=f"{f1:.4f}")

    st.markdown("##### Confusion Matrix:")
    st.dataframe(pd.DataFrame(conf_matrix, index=['Actual Denied', 'Actual Approved'], columns=['Predicted Denied', 'Predicted Approved']))
    st.markdown("##### Classification Report:")
    st.text(classification_report(y_test, y_pred))
    return accuracy, precision, recall, f1, conf_matrix, y_pred

# Section 7: Local Explainability with LIME
def generate_lime_explanation(model, X_train_df, X_test_df, feature_names, instance_idx):
    """
    Generates and visualizes a LIME explanation for a specific instance.
    """
    explainer = lime.lime_tabular.LimeTabularExplainer(
        training_data=X_train_df.values,
        feature_names=feature_names,
        class_names=['Denied', 'Approved'],
        mode='classification',
        random_state=42
    )

    instance = X_test_df.iloc[instance_idx]
    explanation = explainer.explain_instance(
        data_row=instance.values,
        predict_fn=model.predict_proba,
        num_features=10,
        num_samples=5000
    )

    st.subheader(f"LIME Explanation for instance {instance_idx}")
    st.write(f"True Label: {st.session_state.y_test.iloc[instance_idx]}, Predicted: {model.predict(instance.to_frame().T)[0]}")
    st.write("Model prediction probabilities: ", model.predict_proba(instance.to_frame().T)[0])

    fig = explanation.as_pyplot_figure()
    plt.title(f'LIME Explanation for Test Instance {instance_idx}')
    plt.tight_layout()
    return fig

# Section 8: Global Explainability with SHAP
def generate_shap_summary(model, X_train_df, X_test_df, feature_names):
    """
    Generates and visualizes a SHAP summary plot for global feature importance.
    """
    explainer = shap.KernelExplainer(model.predict_proba, X_train_df)
    shap_values = explainer.shap_values(X_test_df)

    st.subheader("SHAP Summary Plot (Global Feature Importance)")
    fig, ax = plt.subplots(figsize=(10, 6))
    shap.summary_plot(shap_values[1], X_test_df, feature_names=feature_names, show=False, plot_size=None, ax=ax)
    plt.title('SHAP Global Feature Importance (Loan Approved)')
    plt.tight_layout()
    return fig, shap_values

# Section 9: SHAP Dependence Plots
def generate_shap_dependence_plot(shap_values, X_data, feature_name, interaction_feature=None):
    """
    Generates a SHAP dependence plot for a specific feature.
    """
    st.subheader(f"SHAP Dependence Plot for '{feature_name}'")
    fig, ax = plt.subplots(figsize=(10, 6))
    shap.dependence_plot(
        ind=feature_name,
        shap_values=shap_values[1],
        features=X_data,
        feature_names=X_data.columns.tolist(),
        interaction_index=interaction_feature,
        show=False, ax=ax
    )
    plt.title(f'SHAP Dependence Plot: {feature_name}')
    plt.tight_layout()
    return fig

# Section 11: Calculating Baseline Fairness Metrics
def calculate_fairness_metrics(model, X_test, y_test, protected_attribute_names, privileged_groups, unprivileged_groups, model_name="Model"):
    """
    Calculates and displays Statistical Parity Difference (SPD) and Equal Opportunity Difference (EOD).
    Returns metrics and a matplotlib figure.
    """
    aif_test_df = pd.concat([X_test, y_test], axis=1)
    aif_test_ds = StandardDataset(
        aif_test_df,
        label_name='LoanApproved',
        favorable_classes=[1],
        protected_attribute_names=['Gender_1'], # Assuming 'Gender_1' is the one-hot encoded male column
        privileged_classes=[[1]],
        unprivileged_classes=[[0]] # Implied for Gender_0 (female)
    )

    y_pred = model.predict(X_test)
    dataset_pred = aif_test_ds.copy()
    dataset_pred.labels = y_pred

    metric = ClassificationMetric(
        aif_test_ds,
        dataset_pred,
        unprivileged_groups=unprivileged_groups,
        privileged_groups=privileged_groups
    )

    spd = metric.statistical_parity_difference()
    eod = metric.equal_opportunity_difference()
    tpr_priv = metric.true_positive_rate(privileged=True)
    tpr_unpriv = metric.true_positive_rate(privileged=False)
    pos_outcome_priv = metric.positive_prediction_rate(privileged=True)
    pos_outcome_unpriv = metric.positive_prediction_rate(privileged=False)

    st.subheader(f"--- {model_name} Fairness Metrics ---")
    st.write(f"Protected Attribute: Gender (Privileged: Male, Unprivileged: Female)")
    st.write(f"Statistical Parity Difference (SPD): `{spd:.4f}`")
    st.write(f"Equal Opportunity Difference (EOD): `{eod:.4f}`")
    st.write(f"True Positive Rate (Privileged Group): `{tpr_priv:.4f}`")
    st.write(f"True Positive Rate (Unprivileged Group): `{tpr_unpriv:.4f}`")
    st.write(f"Positive Outcome Rate (Privileged Group): `{pos_outcome_priv:.4f}`")
    st.write(f"Positive Outcome Rate (Unprivileged Group): `{pos_outcome_unpriv:.4f}`")

    metrics_data = {
        'Metric': ['SPD', 'EOD'],
        'Value': [spd, eod]
    }
    metrics_df = pd.DataFrame(metrics_data)

    fig, ax = plt.subplots(figsize=(8, 5))
    sns.barplot(x='Metric', y='Value', data=metrics_df, palette='viridis', ax=ax)
    ax.axhline(0, color='grey', linestyle='--', linewidth=0.8)
    ax.set_title(f'{model_name} Fairness Metrics')
    ax.set_ylabel('Metric Value')
    ax.set_ylim(-0.5, 0.5)
    plt.tight_layout()
    return spd, eod, fig

# Section 12: Bias Mitigation Technique: Reweighting
def apply_reweighing_and_retrain(X_train, y_train, protected_attribute_names, privileged_groups, unprivileged_groups):
    """
    Applies Reweighing preprocessing and retrains a Logistic Regression model.
    """
    aif_train_df = pd.concat([X_train, y_train], axis=1)
    aif_train_ds = StandardDataset(
        aif_train_df,
        label_name='LoanApproved',
        favorable_classes=[1],
        protected_attribute_names=['Gender_1'],
        privileged_classes=[[1]],
        unprivileged_classes=[[0]]
    )
    RW = Reweighing(unprivileged_groups=unprivileged_groups, privileged_groups=privileged_groups)
    aif_train_ds_reweighed = RW.fit_transform(aif_train_ds)
    X_train_reweighed = pd.DataFrame(aif_train_ds_reweighed.features, columns=X_train.columns, index=X_train.index)
    y_train_reweighed = aif_train_ds_reweighed.labels.ravel()
    sample_weights = aif_train_ds_reweighed.instance_weights
    reweighed_model = train_model(X_train_reweighed, y_train_reweighed, sample_weight=sample_weights)
    return reweighed_model

# Section 14: Bias Mitigation Technique: Threshold Adjustment
def apply_threshold_adjustment_and_evaluate(model, X_train, y_train, X_test, y_test, protected_attribute_names, privileged_groups, unprivileged_groups):
    """
    Applies Calibrated Equalized Odds Postprocessing for threshold adjustment and re-evaluates.
    Returns metrics, adjusted predictions, and a matplotlib figure.
    """
    aif_train_df = pd.concat([X_train, y_train], axis=1)
    aif_train_ds = StandardDataset(
        aif_train_df, label_name='LoanApproved', favorable_classes=[1],
        protected_attribute_names=['Gender_1'], privileged_classes=[[1]], unprivileged_classes=[[0]]
    )
    aif_test_df = pd.concat([X_test, y_test], axis=1)
    aif_test_ds = StandardDataset(
        aif_test_df, label_name='LoanApproved', favorable_classes=[1],
        protected_attribute_names=['Gender_1'], privileged_classes=[[1]], unprivileged_classes=[[0]]
    )
    y_proba_train = model.predict_proba(X_train)[:, 1]
    y_proba_test = model.predict_proba(X_test)[:, 1]
    dataset_pred_train = aif_train_ds.copy(deepcopy=True)
    dataset_pred_train.scores = y_proba_train.reshape(-1, 1)
    dataset_pred_test = aif_test_ds.copy(deepcopy=True)
    dataset_pred_test.scores = y_proba_test.reshape(-1, 1)

    ceopp = CalibratedEqOddsPostprocessing(
        unprivileged_groups=unprivileged_groups, privileged_groups=privileged_groups, cost_constraint='weighted'
    )
    ceopp = ceopp.fit(aif_train_ds, dataset_pred_train)
    dataset_pred_transformed_test = ceopp.predict(dataset_pred_test)
    y_pred_adjusted = dataset_pred_transformed_test.labels.ravel()

    adjusted_metrics = ClassificationMetric(
        aif_test_ds, dataset_pred_transformed_test,
        unprivileged_groups=unprivileged_groups, privileged_groups=privileged_groups
    )
    adjusted_spd = adjusted_metrics.statistical_parity_difference()
    adjusted_eod = adjusted_metrics.equal_opportunity_difference()

    adjusted_accuracy = accuracy_score(y_test, y_pred_adjusted)
    adjusted_precision = precision_score(y_test, y_pred_adjusted)
    adjusted_recall = recall_score(y_test, y_pred_adjusted)
    adjusted_f1 = f1_score(y_test, y_pred_adjusted)
    adjusted_conf_matrix = confusion_matrix(y_test, y_pred_adjusted)

    st.subheader("--- Threshold-Adjusted Model Evaluation ---")
    st.metric(label="Accuracy", value=f"{adjusted_accuracy:.4f}")
    st.metric(label="Precision", value=f"{adjusted_precision:.4f}")
    st.metric(label="Recall", value=f"{adjusted_recall:.4f}")
    st.metric(label="F1-Score", value=f"{adjusted_f1:.4f}")
    st.markdown("##### Confusion Matrix:")
    st.dataframe(pd.DataFrame(adjusted_conf_matrix, index=['Actual Denied', 'Actual Approved'], columns=['Predicted Denied', 'Predicted Approved']))
    st.markdown("##### Classification Report:")
    st.text(classification_report(y_test, y_pred_adjusted))

    st.subheader("--- Threshold-Adjusted Model Fairness Metrics ---")
    st.write(f"Protected Attribute: Gender (Privileged: Male, Unprivileged: Female)")
    st.write(f"Statistical Parity Difference (SPD): `{adjusted_spd:.4f}`")
    st.write(f"Equal Opportunity Difference (EOD): `{adjusted_eod:.4f}`")

    metrics_data = {
        'Metric': ['SPD', 'EOD'],
        'Value': [adjusted_spd, adjusted_eod]
    }
    metrics_df = pd.DataFrame(metrics_data)

    fig, ax = plt.subplots(figsize=(8, 5))
    sns.barplot(x='Metric', y='Value', data=metrics_df, palette='cividis', ax=ax)
    ax.axhline(0, color='grey', linestyle='--', linewidth=0.8)
    ax.set_title(f'Threshold-Adjusted Model Fairness Metrics')
    ax.set_ylabel('Metric Value')
    ax.set_ylim(-0.5, 0.5)
    plt.tight_layout()
    return adjusted_accuracy, adjusted_spd, adjusted_eod, y_pred_adjusted, fig

# Section 15: Comparative Analysis of Mitigation Techniques
def plot_comparative_metrics(baseline_acc, baseline_spd, baseline_eod,
                                 reweighed_acc, reweighed_spd, reweighed_eod,
                                 adjusted_acc, adjusted_spd, adjusted_eod):
    """
    Plots comparative bar charts for accuracy, SPD, and EOD across models.
    """
    metrics = ['Accuracy', 'SPD', 'EOD']
    baseline_values = [baseline_acc, baseline_spd, baseline_eod]
    reweighed_values = [reweighed_acc, reweighed_spd, reweighed_eod]
    adjusted_values = [adjusted_acc, adjusted_spd, adjusted_eod]

    df_comparison = pd.DataFrame({
        'Metric': metrics * 3,
        'Value': baseline_values + reweighed_values + adjusted_values,
        'Model': ['Baseline'] * 3 + ['Reweighed'] * 3 + ['Threshold-Adjusted'] * 3
    })

    fig, axes = plt.subplots(1, 3, figsize=(18, 6), sharey=False)

    sns.barplot(x='Model', y='Value', data=df_comparison[df_comparison['Metric'] == 'Accuracy'], ax=axes[0], palette='pastel')
    axes[0].set_title('Model Accuracy Comparison')
    axes[0].set_ylabel('Accuracy')
    axes[0].set_ylim(0.6, 0.8)

    sns.barplot(x='Model', y='Value', data=df_comparison[df_comparison['Metric'] == 'SPD'], ax=axes[1], palette='deep')
    axes[1].axhline(0, color='grey', linestyle='--', linewidth=0.8)
    axes[1].set_title('Statistical Parity Difference (SPD) Comparison')
    axes[1].set_ylabel('SPD Value')
    axes[1].set_ylim(-0.2, 0.2)

    sns.barplot(x='Model', y='Value', data=df_comparison[df_comparison['Metric'] == 'EOD'], ax=axes[2], palette='dark')
    axes[2].axhline(0, color='grey', linestyle='--', linewidth=0.8)
    axes[2].set_title('Equal Opportunity Difference (EOD) Comparison')
    axes[2].set_ylabel('EOD Value')
    axes[2].set_ylim(-0.2, 0.2)

    plt.tight_layout()
    return fig