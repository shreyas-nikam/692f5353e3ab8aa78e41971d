id: 692f5353e3ab8aa78e41971d_user_guide
summary: AI Design and deployment lab 5 User Guide
feedback link: https://docs.google.com/forms/d/e/1FAIpQLSfWkOK-in_bMMoHSZfcIvAeO58PAH9wrDqcxnJABHaxiDqhSA/viewform?usp=sf_link
environments: Web
status: Published
# AI Credit Decision Explainer: A Guide to Trustworthy AI

## 1. Welcome to the AI Credit Decision Explainer
Duration: 0:03:00

In this lab, you will explore the **AI Credit Decision Explainer**, an interactive application designed to provide comprehensive insights into an AI-driven credit loan application scenario. This tool allows you to simulate training a classification model, understand its decision-making process through local and global explainability techniques, identify and quantify potential biases, and evaluate various bias mitigation strategies.

<aside class="positive">
<b>Why is this important?</b> As AI systems become more prevalent in critical domains like finance, understanding how they make decisions and ensuring they are fair and unbiased is paramount. This application helps demystify these complex topics.
</aside>

### Learning Goals

Upon completing this codelab, you will be able to:

*   **Understand Local Explainability (LIME)**: Learn how LIME reveals the specific factors influencing individual credit decisions, providing transparency for each applicant.
*   **Interpret Global Feature Importance (SHAP)**: Grasp the model's overall decision-making patterns by understanding which features are most influential across the entire dataset.
*   **Identify and Quantify Bias**: Use fairness metrics like Statistical Parity Difference (SPD) and Equal Opportunity Difference (EOD) to detect and measure potential biases in credit decisions across different demographic groups.
*   **Evaluate Bias Mitigation**: Assess the impact of simple mitigation techniques, such as reweighting and classification threshold adjustments, on both fairness and overall model performance.

### Target Audience

This codelab is designed for:

*   **Risk Managers**: To gain insights into how AI models make credit decisions, assess associated risks, and understand the factors driving approvals or denials. This knowledge aids in responsible risk management and compliance.
*   **Executive Stakeholders**: To understand the trustworthiness, fairness, and transparency of AI systems in critical financial applications, enabling informed governance and strategic decision-making.
*   **Financial Data Engineers**: To understand the practical application of explainability and fairness tools, and how to implement basic bias detection and mitigation strategies within their systems.

## 2. Setting Up Your Data
Duration: 0:02:00

Before we can train any models, we need to load and prepare our dataset. This application uses a modified version of the 'UCI German Credit Data' dataset, which contains various demographic and financial attributes relevant to credit decisions. The goal is to predict 'loan approval' or 'loan denial'.

In this step, we will:
*   Load the dataset from a CSV file.
*   Perform initial data preprocessing, including renaming columns, mapping the target variable, creating age groups, and defining 'Gender' and 'Age Group' as protected attributes for later fairness analysis.
*   Split the data into training and testing sets.

<aside class="positive">
<b>Action:</b> Navigate to the "Data Preparation" page using the sidebar and click the "Load and Preprocess Data" button.
</aside>

After clicking, you will see a preview of the processed data, its shape, and confirmation that the data has been successfully prepared. Notice how the original columns have been transformed and new features like 'AgeGroup' and 'Gender' have been created. The app also informs you about the protected attributes identified for fairness analysis.

## 3. Training and Evaluating the Baseline Model
Duration: 0:03:00

With our data prepared, we can now train our first machine learning model, which will serve as our "baseline" for comparison. We will use a Logistic Regression model for this purpose, known for its simplicity and interpretability.

Logistic Regression is a statistical model that estimates the probability of an event occurring, such as a loan being approved. It uses a logistic function to map any real-valued number into a probability between 0 and 1. The core idea is to find a linear combination of features that best predicts the outcome.

The mathematical representation of Logistic Regression for binary classification is:
$$ P(\hat{Y}=1 | X) = \frac{1}{1 + e^{-(\beta_0 + \beta_1X_1 + \dots + \beta_nX_n)}} $$
where $P(\hat{Y}=1 | X)$ is the probability of the positive class (e.g., loan approved), $X$ represents the input features, and $\beta$ are the model's coefficients learned during training.

After training, we will evaluate the model's performance using standard classification metrics:

*   **Accuracy**: The proportion of total predictions that were correct.
*   **Precision**: Out of all predicted approvals, how many were actually approved.
*   **Recall**: Out of all actual approvals, how many did the model correctly identify.
*   **F1-Score**: The harmonic mean of Precision and Recall, providing a single metric that balances both.
*   **Confusion Matrix**: A table that visually summarizes the model's predictions, showing True Positives (correct approvals), True Negatives (correct denials), False Positives (incorrect approvals), and False Negatives (incorrect denials).

<aside class="positive">
<b>Action:</b> Navigate to the "Baseline Model" page in the sidebar. Click the "Train Baseline Model" button.
</aside>

Once the model is trained, you will see its performance metrics and the confusion matrix. Pay attention to these values, as we will compare them against models that incorporate bias mitigation techniques later.

## 4. Understanding Individual Decisions with LIME
Duration: 0:04:00

Now that we have a trained baseline model, it's crucial to understand *why* it makes certain decisions. This is where Explainable AI (XAI) comes into play. We'll start with **LIME (Local Interpretable Model-agnostic Explanations)**, a technique used to explain individual predictions.

LIME works by approximating the complex "black-box" model (like our Logistic Regression) with a simpler, more interpretable model (e.g., a linear model) *locally* around the specific prediction you want to explain. It does this by perturbing the input instance, observing how the black-box model's prediction changes, and then training the simple model on these perturbed samples, weighted by their proximity to the original instance.

The weighted loss function minimized by LIME is:
$$ L(f, g, \pi_x) = \sum_{z \in Z} \pi_x(z) (f(z) - g(z))^2 $$
where $f$ is the black-box model, $g$ is the interpretable model, $Z$ is the set of perturbed samples, and $\pi_x(z)$ is the proximity measure of $z$ to the instance $x$ being explained.

<aside class="positive">
<b>Action:</b> Navigate to the "Explainability" page. Use the slider "Select Test Instance Index for LIME" to choose an applicant from the test set. Then, click "Generate LIME Explanation".
</aside>

A LIME explanation plot will appear, showing which features most influenced the model's decision for that particular individual. Features with positive contributions push the prediction towards 'Approved' (Class 1), while negative contributions push towards 'Denied' (Class 0). This provides valuable insight into the 'right to explanation' for individual loan applicants.

## 5. Unveiling Global Model Behavior with SHAP
Duration: 0:04:00

While LIME helps us understand individual predictions, **SHAP (SHapley Additive exPlanations)** offers a unified framework to explain predictions by assigning an importance value to each feature for *every* prediction. More importantly, it can also provide a global view of feature importance across the entire dataset. SHAP values are based on Shapley values from cooperative game theory, which fairly distribute the "payout" (the prediction) among the "players" (the features).

The SHAP value $\phi_j$ for a feature $j$ is calculated as:
$$ \phi_j = \sum_{S \subseteq N \setminus \{j\}} \frac{|S|!(|N| - |S| - 1)!}{|N|!} [f_x(S \cup \{j\}) - f_x(S)] $$
where $N$ is the set of all features, $S$ is a subset of features not including $j$, and $f_x(S)$ is the model's prediction with features in $S$ present. This formula essentially calculates the average marginal contribution of feature $j$ across all possible combinations of features.

<aside class="positive">
<b>Action:</b> On the "Explainability" page, click "Generate SHAP Summary Plot".
</aside>

The SHAP summary plot provides a global overview of feature importance. Each point represents a Shapley value for a feature for a specific instance in the test set. The color indicates the feature value (red for high, blue for low), and the position on the x-axis indicates the impact on the model's output (higher SHAP value means a higher likelihood of 'Approved'). Features are ordered by their overall impact, helping Risk Managers and Executive Stakeholders understand the primary drivers of credit approval across the entire portfolio.

## 6. Diving Deeper with SHAP Dependence Plots
Duration: 0:03:00

SHAP dependence plots build upon the SHAP values to show how the value of a *single* feature affects the prediction of the model. These plots can reveal non-linear relationships or even interactions with other features. Each point on the plot represents a single prediction from the dataset, much like a scatter plot.

<aside class="positive">
<b>Action:</b> On the "Explainability" page, select a feature from the "Select Feature for Dependence Plot" dropdown. Optionally, select an "Interaction Feature" to see how the primary feature's impact changes based on the value of another feature. Then, click "Generate SHAP Dependence Plot".
</aside>

The SHAP dependence plot illustrates how changes in the selected feature's value correlate with changes in the model's output (probability of loan approval). For example, you might observe that as 'CreditDuration' increases, the model's prediction for approval generally decreases. If an interaction feature is selected, the color coding on the plot will show how the interaction with that second feature further influences the prediction. This gives a more nuanced understanding of specific feature behaviors.

## 7. Assessing Algorithmic Fairness
Duration: 0:04:00

Beyond performance and explainability, evaluating the fairness of an AI model is critical, especially in sensitive areas like credit decisions. Algorithmic bias can lead to discriminatory outcomes for certain demographic groups. We will use two widely accepted fairness metrics from the `aif360` library to quantify these biases:

*   **Statistical Parity Difference (SPD)**: This metric measures the difference in the proportion of positive outcomes (e.g., loan approvals) between the unprivileged group ($a_0$) and the privileged group ($a_1$). A value of 0 indicates perfect statistical parity, meaning both groups have the same approval rate.
    $$ SPD = P(\hat{Y}=1|A=a_1) - P(\hat{Y}=1|A=a_0) $$
    A positive SPD means the privileged group is more likely to receive a positive outcome.

*   **Equal Opportunity Difference (EOD)**: This metric focuses on the true positive rates (recall for the positive class) between the unprivileged group ($a_0$) and the privileged group ($a_1$), specifically among those who are truly qualified (i.e., actual $Y=1$). A value of 0 indicates perfect equal opportunity, meaning both groups are equally likely to be correctly approved if they are creditworthy.
    $$ EOD = P(\hat{Y}=1|A=a_1, Y=1) - P(\hat{Y}=1|A=a_0, Y=1) $$
    A positive EOD means the privileged group is more likely to be correctly approved when they are truly creditworthy.

For our analysis, we will define 'Gender_1' (representing males) as the **privileged group** and 'Gender_0' (representing females) as the **unprivileged group**. The favorable outcome (loan approval) is labeled as 1.

<aside class="positive">
<b>Action:</b> Navigate to the "Fairness Analysis" page. Click the "Calculate Baseline Fairness Metrics" button.
</aside>

The results will show the SPD and EOD for our baseline model. A non-zero value for either metric suggests the presence of bias, with positive values indicating a bias favoring the privileged group. The accompanying bar chart provides a visual representation of these disparities. This step is crucial for identifying if and how our baseline model discriminates.

## 8. Mitigating Bias: Reweighting (Preprocessing)
Duration: 0:04:00

Once bias is detected, the next step is to mitigate it. We'll explore two common techniques, starting with **Reweighting**, a preprocessing bias mitigation technique. This method aims to balance the dataset *before* the model is trained by adjusting the weights of individual training samples.

Reweighting works by assigning different importance (weights) to samples from different groups or with different outcomes during the model training process. For example, instances from an unprivileged group that receive an unfavorable outcome might be given a higher weight. This encourages the model to pay more attention to these samples, effectively reducing disparities in outcomes for protected groups.

The principle involves adjusting sample weights $w_i$ in the loss function during training, typically so that the model minimizes $L = \sum_i w_i \cdot \text{Loss}(y_i, \hat{y}_i)$. By assigning higher weights to instances from unprivileged groups or those with unfavorable outcomes, the model learns to pay more attention to these samples, thus reducing bias.

<aside class="positive">
<b>Action:</b> Navigate to the "Bias Mitigation" page. Click the "Apply Reweighing & Retrain Model" button.
</aside>

After applying reweighting and retraining the model, you will see its new performance metrics and updated fairness metrics (SPD and EOD). Compare these to the baseline model's results. Did reweighting successfully reduce the bias? You might also observe a slight change in overall accuracy, as fairness-accuracy trade-offs are common in bias mitigation.

## 9. Mitigating Bias: Threshold Adjustment (Postprocessing)
Duration: 0:04:00

The second bias mitigation technique we will explore is **Threshold Adjustment**, which is a postprocessing technique. Unlike reweighting, which modifies the training data, threshold adjustment modifies the *classification threshold* ($T$) *after* the model has made its predictions. This technique is applied to the output probabilities of the baseline model.

The core idea is to find optimal, group-specific thresholds that can help achieve fairness targets. For a group $a_0$, the prediction $\hat{Y}=1$ if the predicted probability $P(\hat{Y}=1|A=a_0)$ is greater than its specific threshold $T_{a_0}$. By optimizing these thresholds for different demographic groups, we can equalize metrics like True Positive Rate (to achieve Equal Opportunity) or Positive Prediction Rate (to achieve Statistical Parity) across groups without retraining the model.

<aside class="positive">
<b>Action:</b> On the "Bias Mitigation" page, click the "Apply Threshold Adjustment & Re-evaluate Model" button.
</aside>

The application will now show the performance and fairness metrics for the model after applying threshold adjustment. This technique directly intervenes on the model's output to balance fairness metrics. Observe how this approach affects both accuracy and fairness (SPD, EOD) compared to both the baseline and the reweighted models.

## 10. Comparative Analysis and Key Takeaways
Duration: 0:03:00

This final step brings together all our findings, providing a comparative overview of the baseline model's performance and fairness against the models after applying reweighting and threshold adjustment. This allows for a direct assessment of the trade-offs between model accuracy and fairness improvements achieved by each mitigation strategy.

<aside class="positive">
<b>Action:</b> Navigate to the "Comparative Analysis" page. Click the "Plot Comparative Metrics" button.
</aside>

You will see comparative plots illustrating the impact of each mitigation technique on Accuracy, Statistical Parity Difference (SPD), and Equal Opportunity Difference (EOD).

### Key Takeaways

*   **XAI Tools are Essential**: LIME provides individual-level insights, directly addressing the 'right to explanation', while SHAP reveals global model drivers. Both are indispensable for building trust and transparency.
*   **Fairness Metrics are Crucial**: SPD and EOD are vital for detecting and quantifying algorithmic bias, especially for protected demographic groups. Understanding these metrics is the first step towards responsible AI.
*   **Bias Mitigation Involves Trade-offs**: Both preprocessing (reweighting) and postprocessing (threshold adjustment) techniques can effectively reduce disparities. However, they often involve trade-offs with overall model performance (e.g., a slight decrease in accuracy for improved fairness).
*   **Informed Decision-Making**: Understanding these trade-offs is crucial for responsible AI deployment. Risk Managers and Executive Stakeholders can use this information to balance predictive accuracy with ethical considerations and regulatory compliance, choosing the mitigation strategy that best aligns with their organizational goals and risk appetite.

This concludes your guided tour of the AI Credit Decision Explainer. You have now experienced how to analyze, interpret, and mitigate bias in AI-driven credit decisions.
