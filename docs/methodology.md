# Methodology: Breast Cancer Prediction Engine

This document details the step-by-step methodology used to build the Breast Cancer Prediction Engine, focusing on the data science lifecycle from exploration to deployment.

## 1. Data Collection & Exploration
The project utilizes the **Breast Cancer Wisconsin (Diagnostic) Dataset**.
- **Source**: Digitized images of fine needle aspirates (FNA) of breast masses.
- **Features**: 30 numeric attributes describing cell nuclei characteristics (e.g., radius, texture, perimeter).
- **Target**: Binary classification where 0 represents Malignant and 1 represents Benign.

### Exploratory Data Analysis (EDA)
- **Class Distribution**: The dataset is slightly imbalanced, with ~63% Benign and ~37% Malignant samples.
- **Feature Separation**: Visualizing features using pair plots and box plots revealed that Malignant tumors generally exhibit higher mean values across most geometrical features compared to Benign tumors.
- **Correlation**: A correlation heatmap highlighted significant multicollinearity among size-related features (radius, perimeter, area), which informed our feature selection approach.

## 2. Preprocessing & Feature Engineering
- **Train/Test Split**: Data was split using an 80/20 ratio. We employed **stratified splitting** to ensure the class distribution remained consistent in both sets.
- **Feature Scaling**: Due to the varying scales of the features (e.g., area in hundreds vs. smoothness in decimals), we applied **StandardScaler** to normalize the data, ensuring distance-based algorithms and gradient descent converged properly.
- **Dimensionality Reduction**: Principal Component Analysis (PCA) was applied to reduce the 30 dimensions down to 2 principal components, capturing the majority of the variance and visually confirming that the two classes are linearly separable in a lower-dimensional space.
- **Feature Importance**: A Random Forest model was trained to rank features by their Gini importance. The top features identified were primarily related to the "worst" (largest) measurements of the tumors.

## 3. Model Training
We trained and evaluated five distinct machine learning classifiers:
1. **Logistic Regression**: Served as our strong, interpretable baseline.
2. **Random Forest**: Used for its robustness to outliers and non-linear capabilities.
3. **Support Vector Machine (SVM)**: Applied with an RBF kernel to handle complex decision boundaries.
4. **K-Nearest Neighbors (KNN)**: Evaluated as a simple instance-based learning approach.
5. **Gradient Boosting**: An advanced ensemble method aiming for high accuracy.

## 4. Evaluation
Models were rigorously evaluated using multiple metrics to ensure reliability in a clinical context.
- **Accuracy**: Overall correct predictions.
- **Precision & Recall**: Particular focus was placed on **Recall** for the Malignant class, as minimizing False Negatives (missing a cancer diagnosis) is paramount.
- **ROC-AUC**: The Area Under the Receiver Operating Characteristic curve measured the models' ability to distinguish between classes at various threshold levels.
- **Cross-Validation**: A 5-Fold Stratified Cross-Validation was performed to ensure the models' performance was not dependent on a specific train/test split.

### Final Selection
**Logistic Regression** and **SVM** emerged as the top-performing models, both achieving an impressive 98.25% test accuracy with an F1-Score of 98.61%. Logistic Regression was selected for the final application due to its faster inference time and inherent interpretability.

## 5. Deployment Structure
To bridge the gap between a data science notebook and an engineering product:
- **Refactoring**: Core logic was abstracted into `src/data_handler.py` and `src/model_trainer.py`.
- **Artifacts**: The optimal Logistic Regression model and StandardScaler were exported as `.pkl` files using `joblib`.
- **Web App**: A Streamlit dashboard (`app.py`) was built to consume these artifacts, providing a real-time, user-friendly interface for generating predictions based on custom feature inputs.
