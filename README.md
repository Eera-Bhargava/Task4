# Task4
# Breast Cancer Classification Using Logistic Regression and KNN

This project demonstrates a machine learning pipeline applied to the **Breast Cancer Wisconsin Diagnostic Dataset**. It includes data preprocessing, model training (Logistic Regression and KNN), evaluation using various metrics, and a demonstration of threshold tuning and sigmoid understanding.

---

## What This Project Does

### 1. **Loads and Preprocesses the Data**
- The dataset `breastcancer.csv` is loaded using pandas.
- The `diagnosis` column (categorical: `M` or `B`) is mapped to numerical values: `M = 1` (Malignant), `B = 0` (Benign).
- Irrelevant columns like `id` and `Unnamed: 32` are dropped.
- The dataset is split into features `X` (all columns except diagnosis) and target `y` (the `diagnosis` column).

---

### 2. **User Input for Model Configuration**
- The user inputs:
  - The **test split ratio** (e.g., `0.2` for 80% training and 20% testing).
  - The **number of neighbors** `k` for the KNN Regressor.

---

### 3. **Trains a K-Nearest Neighbors (KNN) Regressor**
- A KNN Regressor is trained on the training data using the distance-based weighting method.
- Predictions are made on the test set.
- Regression metrics such as **Mean Absolute Error (MAE)**, **Mean Squared Error (MSE)**, and **R² score** are printed — even though KNN isn't ideal for classification in this case (this was more for comparison/understanding purposes).

---

### 4. **Standardizes the Features**
- Feature scaling is performed using `StandardScaler()` to bring all input features to the same scale, which is critical for Logistic Regression.
- Both training and test data are scaled.

---

### 5. **Trains a Logistic Regression Model**
- The Logistic Regression model is trained on the scaled training data.
- Predicted probabilities are generated for the test data using `predict_proba()`.

---

### 6. **Evaluates the Classifier**
- A default threshold of 0.5 is applied to convert probabilities to binary class predictions (0 or 1).
- The following metrics are computed:
  - **Confusion Matrix**
  - **Accuracy**
  - **Precision**
  - **Recall**
  - **ROC-AUC Score**
- A **ROC Curve** is plotted to visually represent model performance across different thresholds.

---

### 7. **Performs Threshold Tuning**
- The code manually adjusts the classification threshold (e.g., from 0.5 to 0.4) to show how predictions change.
- This demonstrates how **threshold tuning** impacts recall vs. precision — important in medical decision-making contexts.

---

## Conclusion

This project successfully builds a pipeline to classify breast cancer tumors as malignant or benign. Logistic Regression, when combined with proper scaling and threshold tuning, proves to be a powerful and interpretable model with high ROC-AUC and accuracy.

It also highlights the trade-offs between precision and recall — showing that in medical diagnostics, **false negatives (missing a cancer case)** are often more dangerous than false positives.

Future work can include:
- Cross-validation
- Trying more advanced classifiers like SVM or Random Forest
- Building a front-end (Streamlit or Flask) for deployment

---

