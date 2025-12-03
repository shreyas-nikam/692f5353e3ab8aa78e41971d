id: 692f5353e3ab8aa78e41971d_documentation
summary: AI Design and deployment lab 5 Documentation
feedback link: https://docs.google.com/forms/d/e/1FAIpQLSfWkOK-in_bMMoHSZfcIvAeO58PAH9wrDqcxnJABHaxiDqhSA/viewform?usp=sf_link
environments: Web
status: Published
# Building Responsible AI for Credit Decisions: An Explainer Codelab

## 0. Introduction, Context, and Application Architecture
Duration: 10:00

Welcome to the **AI Credit Decision Explainer** codelab! This interactive guide will walk you through a Streamlit application designed to bring transparency, explainability, and fairness to AI models used in credit loan applications.

### Importance of Responsible AI in Credit Lending

Credit decisions profoundly impact individuals' lives, determining access to financial services, housing, and opportunities. As AI models increasingly automate these decisions, it's paramount to ensure they are not only efficient and accurate but also fair, transparent, and understandable. Unfair or biased AI systems can perpetuate historical inequalities, leading to significant societal and legal consequences. This application provides a hands-on platform to explore these critical aspects.

### Learning Goals

Upon completing this codelab, you will be able to:

*   **Understand Local Explainability (LIME):** Discover how LIME reveals the specific factors influencing individual credit decisions (e.g., why a particular applicant was approved or denied).
*   **Interpret Global Feature Importance (SHAP):** Grasp the model's overall decision-making patterns by identifying the most impactful features across the entire dataset.
*   **Identify Potential Biases:** Calculate and visualize fairness metrics like Statistical Parity Difference (SPD) and Equal Opportunity Difference (EOD) to detect disparities in credit outcomes across different demographic groups.
*   **Evaluate Bias Mitigation Techniques:** Assess the effectiveness of simple mitigation strategies, such as reweighting and classification threshold adjustments, in improving fairness while maintaining model performance.

### Target Audience

This codelab and the underlying application are designed for a diverse group of professionals interested in responsible AI in finance:

*   **Risk Managers**: Gain deep insights into AI model behavior, assess associated risks, and understand the factors driving credit approvals and denials. This tool helps in scrutinizing model decisions for regulatory compliance and internal risk assessment.
*   **Executive Stakeholders**: Understand the trustworthiness, fairness, and transparency of AI systems deployed in critical financial applications. It facilitates informed governance and strategic decision-making regarding AI adoption.
*   **Financial Data Engineers & Scientists**: Learn the practical application of explainability and fairness tools. Explore how to implement basic bias detection and mitigation strategies within a real-world context, enhancing skills in building ethical AI solutions.

### Application Architecture and Data Flow

The Streamlit application is structured into several distinct pages, each focusing on a specific aspect of the AI credit decision lifecycle. The application leverages `st.session_state` to maintain data and model objects across different pages, allowing for a seamless user experience where outputs from one step serve as inputs for the next.

**Conceptual Flow:**

1.  **Home/Overview:** Introduces the application and its objectives.
2.  **Data Preparation:** Loads and preprocesses the German Credit dataset, splitting it into training and testing sets, and applying transformations. The processed data and preprocessor are stored in `st.session_state`.
3.  **Baseline Model Training:** Trains a Logistic Regression model on the prepared training data. The trained model, its predictions, and performance metrics are stored in `st.session_state`.
4.  **Explainability (LIME & SHAP):** Uses the trained baseline model and test data to generate local (LIME) and global (SHAP) explanations.
5.  **Fairness Analysis:** Evaluates the baseline model's predictions for fairness using metrics like SPD and EOD, focusing on a protected attribute (e.g., gender). The fairness metrics are stored in `st.session_state`.
6.  **Bias Mitigation:** Applies pre-processing (Reweighing) and post-processing (Threshold Adjustment) techniques to address identified biases, retraining or re-evaluating the model. Mitigated models and their metrics are stored.
7.  **Comparative Analysis:** Presents a side-by-side comparison of the baseline and mitigated models across performance and fairness metrics.
8.  **Summary & Key Takeaways:** Recapitulates the key learnings and suggests further exploration.

<aside class="positive">
The use of `st.session_state` is critical in Streamlit multi-page applications for persisting data. It acts as a dictionary that retains values across reruns and page changes, ensuring that models and processed data are available throughout the user's session.
</aside>

### Getting Started

To run this application locally, you would typically:

1.  Save the provided code into an `app.py` file and the `application_pages` directory.
2.  Install the required Python packages (e.g., `streamlit`, `pandas`, `scikit-learn`, `matplotlib`, `seaborn`, `shap`, `lime`, `aif360`).
3.  Execute the command: `streamlit run app.py`

This codelab will guide you through interacting with the application's functionalities without needing to set it up yourself, focusing on the concepts and outcomes.

## 1. Data Preparation
Duration: 05:00

This is the initial step where we load and prepare the dataset for our AI credit decision model. Data preprocessing is a crucial step to ensure the quality and suitability of the data for machine learning algorithms.

### Dataset: German Credit Data

We will use the `german_credit.csv` dataset, which is a common benchmark in fairness research. It contains various demographic and financial attributes that could influence credit decisions.

### Preprocessing Steps

The application applies the following preprocessing steps:

1.  **Loading the Dataset:** The data is loaded using `aif360.sklearn.datasets.fetch_german_credit` for consistency with fairness libraries.
2.  **Target and Features Separation:** The `credit` column (where `1` means "Good Credit" and `0` means "Bad Credit") is identified as the target variable, and the rest are features.
3.  **Feature Identification:** Categorical and numerical features are automatically detected.
4.  **Transformation Pipeline:**
    *   **Numerical Scaling:** Numerical features (e.g., `duration_in_month`, `age_years`) are scaled using `StandardScaler`. This transforms the data to have zero mean and unit variance, preventing features with larger values from dominating the learning process.
    *   **Categorical Encoding:** One-Hot Encoding is applied to all categorical features (e.g., `sex`, `housing`, `purpose`). This converts categorical variables into a numerical format suitable for machine learning models. The `sparse_output=False` argument is crucial for compatibility with explainability libraries like SHAP and LIME.
5.  **Data Splitting:** The dataset is split into training (80%) and testing (20%) sets using `train_test_split` with `stratify=y` to maintain the proportion of target classes in both sets.
6.  **Feature Name Preservation:** All feature names after one-hot encoding are preserved for better interpretability in later stages.

After preprocessing, the transformed training and testing feature data (`X_train_df`, `X_test_df`), target labels (`y_train`, `y_test`), the `preprocessor` object, `all_feature_names`, and `aif360` specific group definitions are stored in `st.session_state`.

**Code Snippet: Data Loading and Preprocessing**

```python
# From application_pages/page_1_data_preparation.py

def load_german_credit_data():
    """Loads the German Credit dataset and returns it as a pandas DataFrame."""
    german_dataset = fetch_german_credit(numeric_only=False)
    df = german_dataset.convert_to_dataframe()[0]
    df.columns = df.columns.str.replace(r'[()-]', '', regex=True).str.replace(' ', '_').str.lower()
    return df

def preprocess_data(df):
    """
    Preprocesses the data:
    - Identifies categorical and numerical features.
    - Creates a preprocessing pipeline (StandardScaler for numerical, OneHotEncoder for categorical).
    - Splits data into training and testing sets.
    - Stores preprocessor, feature names, and AIF360 specific group definitions.
    """
    target_column = 'credit'
    X = df.drop(columns=[target_column])
    y = df[target_column]
    
    # Define protected attributes for AIF360 (after one-hot encoding)
    protected_attribute_names_for_aif360_bld = ['sex_male'] 
    aif360_privileged_groups = [{'sex_male': 1}] 
    aif360_unprivileged_groups = [{'sex_male': 0}] 
    
    categorical_features = X.select_dtypes(include=['object', 'category']).columns.tolist()
    numerical_features = X.select_dtypes(include=np.number).columns.tolist()
    
    preprocessor = ColumnTransformer(
        transformers=[
            ('num', StandardScaler(), numerical_features),
            ('cat', OneHotEncoder(handle_unknown='ignore', sparse_output=False), categorical_features)
        ],
        remainder='passthrough'
    )
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
    preprocessor.fit(X_train)
    
    numeric_feature_names_out = numerical_features
    categorical_feature_names_out = preprocessor.named_transformers_['cat'].get_feature_names_out(categorical_features)
    all_feature_names = list(numeric_feature_names_out) + list(categorical_feature_names_out)
    
    X_train_transformed = preprocessor.transform(X_train)
    X_test_transformed = preprocessor.transform(X_test)
    
    X_train_df = pd.DataFrame(X_train_transformed, columns=all_feature_names, index=X_train.index)
    X_test_df = pd.DataFrame(X_test_transformed, columns=all_feature_names, index=X_test.index)
    
    y_train = y_train.reset_index(drop=True)
    y_test = y_test.reset_index(drop=True)
    
    return X_train_df, X_test_df, y_train, y_test, preprocessor, all_feature_names, \
           protected_attribute_names_for_aif360_bld, aif360_privileged_groups, aif360_unprivileged_groups
```

### User Interaction

In the application, you'd click the "Load Default Dataset (`german_credit.csv`)" button to initiate this process. Upon successful completion, you'll see a snapshot of the raw data, its shape, and a summary of the preprocessing steps.

<aside class="positive">
Proper data preprocessing is foundational for reliable machine learning models. Scaling numerical features prevents dominance by those with larger magnitudes, while one-hot encoding categorical features allows models to process them effectively. Stratified splitting ensures that the class distribution of the target variable is maintained in both training and testing sets, which is particularly important for imbalanced datasets.
</aside>

## 2. Baseline Model Training and Evaluation
Duration: 08:00

In this section, we train a **Logistic Regression** model as our baseline. Logistic Regression is a fundamental classification algorithm, often used for binary outcomes, and it models the probability of a certain class or event occurring.

### Mathematical Basis of Logistic Regression

The core of Logistic Regression is the sigmoid function, which squashes any real-valued number into a probability between 0 and 1. The probability $P(Y=1|X)$ that the dependent variable $Y$ is 1 (e.g., "Good Credit") given input features $X$ is modeled as:

$$ P(Y=1|X) = \frac{1}{1 + e^{-(\beta_0 + \beta_1X_1 + ... + \beta_nX_n)}} $$

Where:
*   $P(Y=1|X)$ is the probability of the positive class (e.g., good credit).
*   $e$ is the base of the natural logarithm.
*   $\beta_0$ is the intercept.
*   $\beta_i$ are the coefficients for each feature $X_i$.

The model then predicts the class based on a threshold (typically 0.5): if $P(Y=1|X) \ge 0.5$, predict 1; otherwise, predict 0.

### Training the Model

The `sklearn.linear_model.LogisticRegression` is used for training. We use `solver='liblinear'` which is a good choice for small datasets. The model is trained on `X_train_df` and `y_train` from the previous data preparation step.

**Code Snippet: Model Training**

```python
# From application_pages/page_2_baseline_model.py
# ...
            try:
                baseline_model = LogisticRegression(random_state=42, solver='liblinear')
                baseline_model.fit(st.session_state.X_train_df, st.session_state.y_train)
                st.session_state.baseline_model = baseline_model

                y_pred = baseline_model.predict(st.session_state.X_test_df)
                y_proba = baseline_model.predict_proba(st.session_state.X_test_df)[:, 1]
                
                st.session_state.baseline_predictions = y_pred
                st.session_state.baseline_probabilities = y_proba
                st.success("Baseline Logistic Regression model trained successfully!")
# ...
```

### Model Performance Evaluation

After training, the model's performance is evaluated on the unseen `X_test_df` using various classification metrics:

*   **Accuracy:** The proportion of correctly classified instances out of the total instances.
*   **Precision:** The proportion of positive identifications that were actually correct. Useful when the cost of false positives is high.
*   **Recall:** The proportion of actual positives that were identified correctly. Useful when the cost of false negatives is high.
*   **F1-Score:** The harmonic mean of Precision and Recall, providing a single metric that balances both.
*   **Confusion Matrix:** A table summarizing the performance of a classification algorithm. It shows true positives, true negatives, false positives, and false negatives.
*   **Classification Report:** Provides a comprehensive summary of precision, recall, f1-score, and support for each class.

**Code Snippet: Evaluation Metrics**

```python
# From application_pages/page_2_baseline_model.py
# ...
                accuracy = accuracy_score(st.session_state.y_test, y_pred)
                precision = precision_score(st.session_state.y_test, y_pred)
                recall = recall_score(st.session_state.y_test, y_pred)
                f1 = f1_score(st.session_state.y_test, y_pred)

                st.session_state.baseline_accuracy = accuracy
                st.session_state.baseline_precision = precision
                st.session_state.baseline_recall = recall
                st.session_state.baseline_f1 = f1

                col1, col2, col3, col4 = st.columns(4)
                col1.metric("Accuracy", f"{accuracy:.4f}")
                col2.metric("Precision", f"{precision:.4f}")
                col3.metric("Recall", f"{recall:.4f}")
                col4.metric("F1-Score", f"{f1:.4f}")

                fig_cm = plot_confusion_matrix(st.session_state.y_test, y_pred, "Baseline Model Confusion Matrix")
                st.pyplot(fig_cm)
                plt.close(fig_cm)
                st.text(classification_report(st.session_state.y_test, y_pred))
# ...
```

### User Interaction

You would click the "Train Baseline Model" button. The application then displays the calculated metrics and a visual confusion matrix, giving you a clear picture of the model's performance. The trained model and its predictions are saved to `st.session_state` for subsequent steps.

<aside class="negative">
A crucial step for building trustworthy AI is establishing a strong baseline. While Logistic Regression is interpretable to some extent (due to its coefficients), for complex models, metrics alone don't explain *why* a decision was made. This is where explainability techniques become vital.
</aside>

## 3. Explainability (LIME & SHAP)
Duration: 15:00

Understanding *why* an AI model makes a particular decision is crucial for trust and transparency, especially in sensitive domains like credit lending. This section explores two powerful explainability techniques: **LIME (Local Interpretable Model-agnostic Explanations)** for individual predictions and **SHAP (SHapley Additive exPlanations)** for global model behavior.

### Local Explainability: LIME

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

### User Interaction: LIME

You can select a test instance index using a slider. The application will then display the instance's true label, predicted label, and predicted probability. Clicking "Generate LIME Explanation" triggers the LIME explainer.

**Code Snippet: LIME Explanation Generation**

```python
# From application_pages/page_3_explainability.py
# ...
                explainer = lime.lime_tabular.LimeTabularExplainer(
                    training_data=st.session_state.X_train_df.values,
                    feature_names=all_feature_names,
                    class_names=['Bad Credit (0)', 'Good Credit (1)'],
                    mode='classification'
                )

                explanation = explainer.explain_instance(
                    data_row=X_test_df.iloc[selected_instance_idx].values,
                    predict_fn=baseline_model.predict_proba,
                    num_features=10 # Explain top 10 features
                )
                
                st.session_state.lime_explanation = explanation

                fig_lime = plot_lime_explanation(explanation, f"LIME Explanation for Instance {selected_instance_idx}")
                st.pyplot(fig_lime)
                plt.close(fig_lime)
# ...
```

The LIME plot shows the contribution of each feature to the model's prediction for that specific instance. Positive values (green) push towards "Good Credit", while negative values (red) push towards "Bad Credit".

### Global Explainability: SHAP (SHapley Additive exPlanations)

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

### SHAP Plots for Global Understanding

SHAP provides two main types of plots for global understanding:

*   **Summary Plot:** Shows the overall impact of features on the model output. Features are ordered by importance. The color represents the feature value (e.g., red for high, blue for low), indicating how high/low values affect the prediction. A high SHAP value (to the right) means the feature is pushing the prediction towards 'Good Credit'.
*   **Dependence Plot:** Visualizes the effect of a single feature on the prediction and how it interacts with another feature. Each point is an instance. The X-axis shows the feature value, and the Y-axis shows its SHAP value. The color can represent an interaction feature.

### User Interaction: SHAP

The SHAP explainer is initialized once using the `LinearExplainer` for our Logistic Regression model. Then, SHAP values are calculated for the entire test set.

**Code Snippet: SHAP Explainer Initialization and Summary Plot**

```python
# From application_pages/page_3_explainability.py
# ...
    if "shap_explainer_baseline" not in st.session_state:
        st.session_state.shap_explainer_baseline = shap.LinearExplainer(
            model=baseline_model,
            data=st.session_state.X_train_df # Use training data for background distribution
        )
        st.session_state.shap_values_baseline = st.session_state.shap_explainer_baseline.shap_values(st.session_state.X_test_df)
# ...
        fig_shap_summary = plot_shap_summary(st.session_state.shap_values_baseline, all_feature_names, "SHAP Summary Plot for Baseline Model")
        st.pyplot(fig_shap_summary)
        plt.close(fig_shap_summary)
# ...
```

For the **Dependence Plot**, you can select a primary feature and an optional interaction feature. Clicking "Generate SHAP Dependence Plot" will display how the selected feature impacts predictions and its interaction with another feature.

<aside class="positive">
LIME and SHAP provide complementary views. LIME offers deep insight into *why* a single decision was made, which is valuable for auditing and compliance. SHAP gives a broad overview of *how* the model behaves generally, aiding in model debugging and understanding systemic behaviors.
</aside>

## 4. Fairness Analysis
Duration: 10:00

Fairness in AI is about ensuring that models do not discriminate against certain demographic groups. This section evaluates the fairness of our baseline model using widely recognized metrics from the `AIF360` library. We will focus on the 'sex' attribute (specifically `sex_male` after one-hot encoding) as our protected attribute for this analysis.

### Key Fairness Metrics

#### 1. Statistical Parity Difference (SPD)
The **Statistical Parity Difference** measures the difference in the proportion of favorable outcomes (e.g., credit approval) between unprivileged and privileged groups. A model achieves statistical parity if the proportion of favorable outcomes is equal across all groups.

$$ \text{SPD} = P(\hat{Y}=1 \mid \text{unprivileged}) - P(\hat{Y}=1 \mid \text{privileged}) $$

Where:
*   $P(\hat{Y}=1 \mid \text{unprivileged})$ is the probability of a favorable prediction ($\hat{Y}=1$) for the unprivileged group.
*   $P(\hat{Y}=1 \mid \text{privileged})$ is the probability of a favorable prediction for the privileged group.

An SPD value of **0** indicates perfect statistical parity. Values close to 0 (both positive and negative) are generally desired.
*   A positive SPD means the unprivileged group receives favorable outcomes more often.
*   A negative SPD means the privileged group receives favorable outcomes more often.

#### 2. Equal Opportunity Difference (EOD)
The **Equal Opportunity Difference** measures the difference in true positive rates (recall) between unprivileged and privileged groups. It focuses on whether groups that *should* receive a favorable outcome (true positives) are equally likely to actually receive it.

$$ \text{EOD} = P(\hat{Y}=1 \mid Y=1, \text{unprivileged}) - P(\hat{Y}=1 \mid Y=1, \text{privileged}) $$

Where:
*   $P(\hat{Y}=1 \mid Y=1, \text{unprivileged})$ is the true positive rate for the unprivileged group (i.e., recall for unprivileged group).
*   $P(\hat{Y}=1 \mid Y=1, \text{privileged})$ is the true positive rate for the privileged group.

An EOD value of **0** indicates equal opportunity. Values close to 0 (both positive and negative) are desired.
*   A positive EOD means the unprivileged group has a higher true positive rate.
*   A negative EOD suggests that the privileged group has a higher chance of getting a 'Good Credit' prediction when they actually have good credit.

### User Interaction and AIF360

To calculate these metrics, the application first constructs `BinaryLabelDataset` objects from the `X_test_df`, `y_test`, and baseline predictions, specifying the `protected_attribute_names`, `privileged_groups`, and `unprivileged_groups` (defined during data preparation).

**Code Snippet: Creating AIF360 Dataset and Calculating Metrics**

```python
# From application_pages/page_4_fairness_analysis.py
# ...
def create_aif360_dataset(X_df, y_series, protected_attribute_names, privileged_groups, unprivileged_groups, feature_names):
    df_aif = X_df.copy()
    df_aif['credit'] = y_series.values
    bld = BinaryLabelDataset(
        df=df_aif,
        label_names=['credit'],
        protected_attribute_names=protected_attribute_names,
        favorable_label=1,
        unfavorable_label=0
    )
    return bld

def calculate_fairness_metrics(dataset_true, dataset_pred, privileged_groups, unprivileged_groups):
    metric = ClassificationMetric(
        dataset_true,
        dataset_pred,
        unprivileged_groups=unprivileged_groups,
        privileged_groups=privileged_groups
    )
    spd = metric.statistical_parity_difference()
    eod = metric.equal_opportunity_difference()
    return spd, eod
# ...
    if st.button("Calculate Fairness Metrics for Baseline Model"):
        # ... (create dataset_true and dataset_pred) ...
        st.session_state.baseline_spd, st.session_state.baseline_eod = calculate_fairness_metrics(
            dataset_true,
            dataset_pred,
            st.session_state.aif360_privileged_groups,
            st.session_state.aif360_unprivileged_groups
        )
# ...
```

Clicking "Calculate Fairness Metrics for Baseline Model" computes and displays the SPD and EOD values, along with a bar chart for visual comparison. These metrics are stored in `st.session_state`.

<aside class="negative">
Understanding the sign and magnitude of SPD and EOD is crucial. A non-zero value indicates bias, and the sign tells you which group is disadvantaged for that specific metric. Often, a model performs well on overall accuracy but exhibits significant bias, making these metrics essential.
</aside>

## 5. Bias Mitigation
Duration: 12:00

Bias mitigation techniques aim to reduce or eliminate unfairness in machine learning models. This section explores two simple, yet effective, mitigation strategies: **Reweighing** (a pre-processing technique) and **Classification Threshold Adjustment** (a post-processing technique).

### Bias Mitigation Strategies

#### 1. Reweighing (Pre-processing technique)
**Reweighing** is a pre-processing technique that assigns different weights to individual training examples based on their protected attributes and labels. This rebalancing helps to equalize the representation of different demographic groups in the training data, leading to a fairer model.

The weights are designed such that the joint probability distribution of the protected attribute and the label becomes fair. For example, if a privileged group with a favorable outcome is over-represented, their samples will be down-weighted during training.

**User Interaction and Implementation:**
Clicking "Apply Reweighing & Retrain Model" first creates an AIF360 `BinaryLabelDataset` from the training data. Then, the `Reweighing` algorithm calculates sample weights, which are used to train a new Logistic Regression model.

**Code Snippet: Reweighing and Retraining**

```python
# From application_pages/page_5_bias_mitigation.py
# ...
    if st.button("Apply Reweighing & Retrain Model"):
        dataset_orig_train = create_aif360_dataset_for_mitigation( # Helper adapted for mitigation page
            st.session_state.X_train_df, st.session_state.y_train,
            st.session_state.protected_attribute_names,
            st.session_state.aif360_privileged_groups, st.session_state.aif360_unprivileged_groups
        )

        RW = Reweighing(
            unprivileged_groups=st.session_state.aif360_unprivileged_groups,
            privileged_groups=st.session_state.aif360_privileged_groups
        )
        dataset_transf_train = RW.fit_transform(dataset_orig_train)
        sample_weights = dataset_transf_train.instance_weights

        reweighed_model = LogisticRegression(random_state=42, solver='liblinear')
        reweighed_model.fit(st.session_state.X_train_df, st.session_state.y_train, sample_weight=sample_weights)
        st.session_state.reweighed_model = reweighed_model
        # ... calculate metrics and store in session_state ...
# ...
```

The performance (accuracy) and fairness metrics (SPD, EOD) for the reweighed model are calculated and displayed.

#### 2. Classification Threshold Adjustment (Post-processing technique)
**Classification Threshold Adjustment** is a post-processing technique that modifies the classification threshold (typically 0.5 for logistic regression) to achieve fairness goals. Instead of using a single threshold for all groups, different thresholds can be applied to different demographic groups to balance fairness and performance. For instance, if the unprivileged group has a lower predicted probability for a favorable outcome, its threshold can be lowered to increase their favorable outcome rate.

**User Interaction and Implementation:**
Clicking "Apply Threshold Adjustment & Re-evaluate Model" takes the predictions (probabilities) from the **baseline model** and searches for an optimized global threshold that improves fairness (specifically, reduces the absolute SPD). It then re-evaluates the baseline model's predictions using this new threshold.

**Code Snippet: Threshold Adjustment**

```python
# From application_pages/page_5_bias_mitigation.py
# ...
    if st.button("Apply Threshold Adjustment & Re-evaluate Model"):
        y_proba_baseline = st.session_state.baseline_probabilities
        dataset_true_test = create_aif360_dataset_for_mitigation( # Helper function
            st.session_state.X_test_df, st.session_state.y_test,
            st.session_state.protected_attribute_names,
            st.session_state.aif360_privileged_groups, st.session_state.aif360_unprivileged_groups
        )
        
        best_threshold = 0.5
        best_spd_abs = abs(st.session_state.baseline_spd) if st.session_state.baseline_spd is not None else 1.0
        
        # Simple grid search for threshold
        thresholds = np.linspace(0.01, 0.99, 50)
        for thresh in thresholds:
            y_pred_adjusted_temp = (y_proba_baseline >= thresh).astype(int)
            dataset_pred_temp = dataset_true_test.copy()
            dataset_pred_temp.labels = y_pred_adjusted_temp.reshape(-1, 1)
            temp_spd, _ = calculate_fairness_metrics(
                dataset_true_test, dataset_pred_temp,
                st.session_state.aif360_privileged_groups, st.session_state.aif360_unprivileged_groups
            )
            if abs(temp_spd) < best_spd_abs:
                best_spd_abs = abs(temp_spd)
                best_threshold = thresh

        st.session_state.threshold_adjusted_predictions = (y_proba_baseline >= best_threshold).astype(int)
        # ... calculate adjusted_accuracy, adjusted_spd, adjusted_eod and store in session_state ...
# ...
```

The performance and fairness metrics for the threshold-adjusted model are then displayed.

<aside class="positive">
Pre-processing techniques like Reweighing intervene at the data level, potentially leading to a fundamentally fairer model. Post-processing techniques like Threshold Adjustment are flexible and can be applied after a model is trained or even deployed, without altering the model itself.
</aside>

## 6. Comparative Analysis of Models
Duration: 07:00

This section provides a comparative view of the **Baseline Model**, the **Reweighed Model**, and the **Threshold-Adjusted Model**. We will compare their performance (Accuracy) and fairness (Statistical Parity Difference and Equal Opportunity Difference) to understand the trade-offs involved in bias mitigation.

### User Interaction

This page automatically retrieves all relevant metrics (Accuracy, SPD, EOD) from `st.session_state` that were calculated in the previous steps. If any metrics are missing, a warning will be displayed, prompting the user to complete the preceding steps.

### Visual Comparison

The metrics are presented in a tabular format, followed by bar charts for visual comparison:

*   **Model Accuracy Comparison:** Shows how the mitigation techniques affected the overall prediction accuracy.
*   **Statistical Parity Difference Comparison (SPD):** Illustrates how each model performs in achieving equal favorable outcome rates across groups. Values closer to 0 are better.
*   **Equal Opportunity Difference Comparison (EOD):** Compares how well each model ensures equal true positive rates for different groups. Values closer to 0 are better.

**Code Snippet: Plotting Comparative Metrics**

```python
# From application_pages/page_6_comparative_analysis.py
# ...
def plot_comparative_metrics(metrics_values, labels, metric_name, title):
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
# ...
    # Example for Accuracy
    fig_accuracy = plot_comparative_metrics(accuracies, labels, 'Accuracy', 'Model Accuracy Comparison')
    st.pyplot(fig_accuracy)
    plt.close(fig_accuracy)
# ...
```

### Interpretation of Results

*   Observe how bias mitigation techniques impact both model performance (accuracy) and fairness metrics (SPD, EOD).
*   Often, there's a **trade-off** between maximizing accuracy and achieving perfect fairness. Mitigating bias might lead to a slight decrease in overall accuracy, but it results in a more equitable model.
*   **Reweighing** (a pre-processing method) adjusts the training data, affecting the model from its core.
*   **Threshold Adjustment** (a post-processing method) adjusts predictions without retraining the model, offering a lighter touch.

By comparing these metrics, you can evaluate which mitigation strategy best aligns with the desired balance between performance and fairness objectives for your specific application.

<aside class="positive">
Comparative analysis is key to making informed decisions. It helps stakeholders understand the implications of different fairness interventions, allowing them to choose a strategy that balances ethical considerations with business objectives.
</aside>

## 7. Summary & Key Takeaways
Duration: 03:00

Congratulations! You have navigated through the **AI Credit Decision Explainer** application and completed this codelab. This journey has provided a hands-on experience in understanding, evaluating, and mitigating biases in AI-driven credit lending.

### Key Learnings from this Lab:

1.  **The Importance of Transparency (Explainability):**
    *   You've seen how **LIME** can pinpoint the exact features influencing an *individual's* credit decision, bringing clarity to "black-box" predictions. This is crucial for regulatory compliance and building trust with applicants.
    *   **SHAP** values offered a *global* perspective, revealing which features generally drive the model's decisions and how feature values impact outcomes across the entire dataset. This helps in understanding the model's overall logic and potential systemic issues.

2.  **Identifying and Quantifying Bias (Fairness Analysis):**
    *   You utilized **Statistical Parity Difference (SPD)** to measure if different demographic groups receive favorable credit outcomes at equal rates. A non-zero SPD indicates a disparity in positive prediction rates.
    *   **Equal Opportunity Difference (EOD)** allowed you to assess if groups that *truly* deserve credit (actual good credit) are equally likely to be predicted as such. A non-zero EOD suggests unequal treatment for equally qualified individuals.
    *   These metrics highlight that models, even when trained on seemingly neutral data, can perpetuate or amplify existing societal biases.

3.  **Strategies for Bias Mitigation:**
    *   **Reweighing (Pre-processing):** You implemented a technique that rebalances the training data by assigning different weights to samples. This aims to create a more balanced representation, leading to a fairer model at the training stage.
    *   **Classification Threshold Adjustment (Post-processing):** You experimented with modifying the decision threshold of the model to achieve fairness goals without retraining. This is a flexible approach that can be applied after a model is deployed.

4.  **The Performance-Fairness Trade-off:**
    *   Through comparative analysis, you observed that improving fairness often comes with a trade-off in overall model performance (e.g., a slight decrease in accuracy). The challenge lies in finding an optimal balance that meets both business objectives and ethical standards.
    *   There is no "one-size-fits-all" solution for fairness. The choice of fairness metric and mitigation technique depends on the specific context, legal requirements, and ethical considerations of the application.

### Moving Forward

The field of Responsible AI is continuously evolving. This lab serves as a foundational step. Consider exploring:
*   More advanced explainability techniques.
*   Other fairness definitions (e.g., disparate impact, predictive equality).
*   More sophisticated in-processing and post-processing bias mitigation algorithms.
*   The ethical implications of AI deployment in various sensitive domains.

Building AI systems that are accurate, transparent, and fair is not just a technical challenge but an ethical imperative. By applying the concepts learned here, you can contribute to developing more responsible and trustworthy AI solutions.
