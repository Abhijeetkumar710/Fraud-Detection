# Fraud-Detection
#  Fraud Detection with Machine Learning

## Project Overview
This project develops a complete machine learning pipeline to detect fraudulent transactions.  
Fraud detection is critical in finance and e-commerce, where fraudulent activities are rare but highly impactful.  
The dataset contains anonymized transaction features, and the goal is to classify whether a transaction is **fraudulent (1)** or **genuine (0)**.

---

## Project Workflow

### **Phase 1: Data Understanding & Preparation**
- Loaded transaction dataset and explored structure.
- Checked class imbalance (frauds are extremely rare).
- Handled missing values and standardized features.
- Split into training and testing sets.

### **Phase 2: Exploratory Data Analysis (EDA)**
- Distribution of fraudulent vs non-fraudulent transactions.
- Feature correlations using heatmap.
- Visual inspection of selected feature distributions.

### **Phase 3: Preprocessing & Balancing**
- Standardized numerical features.
- Handled class imbalance using **SMOTE (Synthetic Minority Oversampling Technique)**.
- Ensured train-test stratification to preserve fraud ratios.

### **Phase 4: Model Training & Evaluation**
- Trained and compared multiple models:
  - Logistic Regression
  - Decision Tree
  - Random Forest
  - Gradient Boosting
- Applied **cross-validation** for robust performance metrics.
- Used precision, recall, F1-score, and ROC-AUC to handle class imbalance effectively.

### **Phase 5: Model Saving & Inference**
- Saved final model, scaler, and feature list using `joblib`.
- Built an inference pipeline for predicting fraud on new transactions.
- Example transaction was successfully classified as **fraudulent/genuine**.

---

## Results Summary

| Model              | CV Accuracy (± Std) | Test Accuracy | ROC-AUC |
|--------------------|----------------------|---------------|---------|
| Logistic Regression | ~0.94 ± 0.01        | ~0.95         | ~0.96   |
| Decision Tree       | ~0.91 ± 0.02        | ~0.92         | ~0.93   |
| Random Forest       | ~0.98 ± 0.01        | ~0.98         | ~0.99   |
| Gradient Boosting   | ~0.97 ± 0.01        | ~0.97         | ~0.98   |

- Final model: **Random Forest Classifier**  
- Balanced performance between precision and recall.  
- Robust against class imbalance after applying SMOTE.  

---

##  Conclusion
- Built an end-to-end ML pipeline to detect fraud.  
- Handled severe **class imbalance** using SMOTE.  
- **Random Forest** chosen as final model due to its robustness and high ROC-AUC.  
- In real-world deployment, this helps financial institutions prevent fraud while minimizing false positives.  

---

## Tech Stack
- Python, Pandas, NumPy
- Scikit-Learn
- Matplotlib, Seaborn
- Imbalanced-learn (SMOTE)
- Joblib (model persistence)

---

## 📂 Repository Structure
