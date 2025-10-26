## 🩺 Breast Cancer Prediction Using SVM with PCA

### 📘 Project Overview

This project implements a machine learning pipeline to predict Breast Cancer diagnosis (Benign or Malignant) using the Wisconsin Diagnostic Breast Cancer (WDBC) dataset.

It uses Support Vector Machine (SVM) with PCA for dimensionality reduction, combined with robust preprocessing (Yeo-Johnson Transformation and Standard Scaling).
A Streamlit web app is deployed for real-time predictions based on user input.

### 🚀 Live App

👉 Try the App Here: https://breastcancer-pca.streamlit.app/

<img src="https://github.com/user-attachments/assets/3f0d11f5-e957-43a1-a571-0cca21f9b98a" alt="App Screenshot" width="300">

<img src="https://github.com/user-attachments/assets/5a3a870b-496a-4d79-97b0-e63771fae621" alt="App Screenshot" width="300">

### 📂 Dataset

Source: UCI Machine Learning Repository – Breast Cancer Wisconsin (Diagnostic)

Link - https://archive.ics.uci.edu/dataset/17/breast+cancer+wisconsin+diagnostic

Filename used: wdbc.data

This dataset contains 569 records and 32 columns (30 features + ID + Diagnosis).

### 🧠 Project Structure

BreastCancerPrediction/
│
├── wdbc.data                  # Original dataset
├── notebook.ipynb             # Jupyter Notebook for EDA, preprocessing, and model training
├── pt.pkl                     # Saved PowerTransformer object
├── scaler.pkl                 # Saved StandardScaler object
├── pca.pkl                    # Saved PCA object
├── svm.pkl                    # Trained SVM model
├── breast_cancer_svm.pkl      # Complete pipeline including preprocessing + PCA + SVM
├── app.py                     # Streamlit app for prediction
└── README.md                  # Project documentation

### 🧩 Libraries and Tools

Python 3.8+

pandas

numpy

matplotlib

seaborn

scikit-learn

streamlit

pickle (for saving/loading models)

### 👉 Steps Performed

#### 1. Data Loading & Inspection

Loaded wdbc.data with column names.

Checked for nulls and duplicates.

Mapped Diagnosis to 0 = Benign, 1 = Malignant.

#### 2. Exploratory Data Analysis (EDA)

Visualized diagnosis distribution.

Analyzed correlations to identify top features.

Checked feature skewness and visualized distributions.

Boxplots for outlier detection.

#### 3. Data Preprocessing

Applied Yeo-Johnson Power Transformation to skewed features.

Clipped outliers using the IQR method.

Scaled features using StandardScaler.

#### 4. Dimensionality Reduction

Applied PCA to reduce features to 6 components (~90% explained variance).

#### 5. Model Training

Split data into training and testing sets (80:20).

Trained SVM with RBF kernel.

Evaluated accuracy: ~98.2%.

Confusion matrix and classification report generated.

#### 6. Saving Objects

Saved preprocessing objects and trained model separately:
pt.pkl, scaler.pkl, pca.pkl, svm.pkl.

Also created complete pipeline combining preprocessing, scaling, PCA, and SVM: breast_cancer_svm.pkl.

### ⚙️ How to Run the Project

#### 1. Clone Repository

git clone <your-repo-link>
cd BreastCancerPrediction

#### 2. Install Dependencies

pip install pandas numpy matplotlib seaborn scikit-learn streamlit

#### 3. Run Jupyter Notebook (Optional)

jupyter notebook notebook.ipynb

#### 4. Run Streamlit App

streamlit run app.py

### ✅ Accuracy & Evaluation

| Metric                    | Value  |
| :------------------------ | :----- |
| **Accuracy**              | 98.24% |
| **Precision (Malignant)** | 1.00   |
| **Recall (Malignant)**    | 0.95   |
| **F1-score (Malignant)**  | 0.98   |

The model is robust and performs well for both Benign and Malignant classes.

### 📊 Confusion Matrix

| Actual / Predicted | Benign | Malignant |
| ------------------ | :----: | :-------: |
| **Benign**         |   72   |     0     |
| **Malignant**      |    2   |     40    |

### 🌐 Streamlit App Overview

#### Features:

Enter all 30 numeric feature values.

Click “🎯 Predict” to get:

Prediction result (Benign / Malignant)

Class probabilities

Clean, user-friendly interface with color-coded output.

### Notes

Ensure that the same preprocessing (PowerTransformer + StandardScaler + PCA) is applied in the Streamlit app as during training.
