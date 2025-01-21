# Breast Cancer Detection with Machine Learning

This project aims to detect breast cancer (malignant or benign) using machine learning algorithms. We perform multiple model evaluations, handle class imbalance using SMOTE, and visualize model performance with various metrics. Model interpretability is also achieved using SHAP values.

## Table of Contents
- [Introduction](#introduction)
- [Dataset](#dataset)
- [Technologies Used](#technologies-used)
- [Steps Involved](#steps-involved)
- [Model Performance](#model-performance)
- [Model Interpretability](#model-interpretability)
- [How to Run the Project](#how-to-run-the-project)
- [Contributing](#contributing)

## Introduction

This project classifies breast cancer into two categories: **Malignant** and **Benign**. We explore various machine learning models, preprocess the data to handle class imbalance, and evaluate the models based on precision, recall, and F1-score. The top-performing model is selected based on these metrics.

## Dataset

The dataset used for this project is the **Breast Cancer Wisconsin (Diagnostic) Dataset** from the UCI Machine Learning Repository. It consists of features computed from breast cancer digitized images. The features describe the characteristics of the cell nuclei present in the images.

- **Link to dataset**: [UCI Breast Cancer Dataset](https://archive.ics.uci.edu/ml/machine-learning-databases/breast-cancer-wisconsin/wdbc.data)

### Data Columns:
- `ID`: The ID of the sample
- `Diagnosis`: The target variable (Malignant = M, Benign = B)
- `Feature_1 to Feature_30`: 30 feature columns describing characteristics of the cell nuclei

## Technologies Used

- **Python Libraries**:
  - `numpy`, `pandas`: Data manipulation
  - `matplotlib`, `seaborn`: Data visualization
  - `scikit-learn`: Machine learning models and metrics
  - `xgboost`: Gradient boosting algorithm
  - `imbalanced-learn`: For handling class imbalance (SMOTE, ADASYN)
  - `shap`: Model interpretability and feature importance
- **Jupyter Notebook**: For developing and running the machine learning pipeline

## Steps Involved

1. **Data Loading**: We load the Breast Cancer dataset and clean the data by removing unnecessary columns and mapping the target variable (`Diagnosis`) to binary values.
2. **Data Preprocessing**: We split the dataset into features (X) and target (y), then split them into training and testing sets. We also standardize the features.
3. **SMOTE for Class Imbalance**: To handle class imbalance, we use ADASYN (a variant of SMOTE) to oversample the minority class in the training dataset.
4. **Model Definition**: We define multiple machine learning models like Logistic Regression, SVM, Random Forest, Gradient Boosting, XGBoost, and others.
5. **Model Evaluation**: We evaluate each model using precision, recall, and F1-score and select the best model based on these metrics.
6. **Model Interpretability**: We use SHAP to explain the model's predictions and visualize feature importance.

## Model Performance

The models were evaluated on:
- **Precision**: How many selected items are relevant.
- **Recall**: How many relevant items are selected.
- **F1-Score**: The weighted average of Precision and Recall.

The top-performing model achieved:
- **Precision**: 0.98
- **Recall**: 0.98
- **F1-Score**: 0.98

## Model Interpretability

We used **SHAP** (SHapley Additive exPlanations) to:
- Visualize the impact of each feature on model predictions.
- Create summary plots and force plots to interpret individual predictions.

## How to Run the Project

1. Clone this repository:
git clone https://github.com/your-username/breast-cancer-detection.git cd breast-cancer-detection

2. Install the required dependencies:
pip install -r requirements.txt

3. Run the Jupyter Notebook:
jupyter notebook breast_cancer_detection.ipynb


## Contributing

Feel free to fork this repository and contribute to it. You can:
- Suggest improvements
- Open an issue for bug reports or feature requests
- Submit a pull request for proposed changes

---

### License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.



