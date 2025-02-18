# Heart Disease Prediction  

This project is a **machine learning-based heart disease prediction system** built using **Python, XGBoost, and Streamlit**. It processes heart disease dataset features, trains a classification model, and provides an interactive UI for users to input their health parameters and receive predictions.

## Table of Contents  
- Overview
- Key Features
- Dataset
- Installation
- Usage
- Model Training
- Web Application
- Resultes
---

## Overview  
This project analyzes a heart disease dataset using **exploratory data analysis (EDA)** and builds an **XGBoost-based classifier** to predict heart disease based on patient data. The model achieves cross-validation and test accuracy scores and is deployed via a **Streamlit web application**.  

---

### Key Features:
- Exploratory Data Analysis (EDA) with **Pandas, NumPy, and Plotly**  
- **Feature engineering** and preprocessing  
- **XGBoost classifier** with cross-validation  
- **Streamlit web app** for user-friendly predictions  
- **Pickle serialization** for model deployment  

---

## Dataset  
The dataset used is **heart.csv**, which contains patient health metrics like age, cholesterol levels, blood pressure, and heart disease presence (binary classification: 0 - No Disease, 1 - Disease).  

### Columns:
- **Age**: Patient's age  
- **Sex**: Male (M) / Female (F)  
- **ChestPainType**: Types of chest pain (ASY, NAP, ATA, TA)  
- **RestingBP**: Resting blood pressure (mm Hg)  
- **Cholesterol**: Serum cholesterol (mg/dl)  
- **FastingBS**: Fasting blood sugar (>120 mg/dl: 1, else 0)  
- **RestingECG**: Resting ECG results (Normal, LVH, ST)  
- **MaxHR**: Maximum heart rate achieved  
- **ExerciseAngina**: Exercise-induced angina (Yes/No)  
- **Oldpeak**: ST depression induced by exercise  
- **ST_Slope**: Slope of the ST segment (Up, Flat, Down)  
- **HeartDisease**: Target variable (0: No disease, 1: Heart disease)  

---

## Installation  
To set up the project locally, follow these steps:

### 1️⃣ Clone the Repository  
```bash
git clone https://github.com/Ahmed-Ramadan-Ahmed/heart-disease-prediction.git
cd heart-disease-prediction
```

### 2️⃣ Install Dependencies
Ensure you have Python 3.7+ installed. Then install the required packages:

```bash
pip install -r requirements.txt
```

## Usage
### Run the Streamlit Web App
```bash
streamlit run app.py
```

This will start a local web server where you can input health parameters and receive predictions.

---

## Model Training  
The model is trained using **XGBoost** with the following pipeline:

### 1️⃣ Data Preprocessing  
- Convert categorical variables using **one-hot encoding**  
- Normalize numerical features  
- Handle missing values  

### 2️⃣ Splitting the Dataset  
- **80% training**, **20% testing**  

### 3️⃣ Training the XGBoost Classifier  

```python
from xgboost import XGBClassifier

model = XGBClassifier(
    objective="binary:logistic",
    subsample=0.7,
    min_child_weight=7,
    max_depth=3,
    learning_rate=0.1,
    gamma=0.0,
    colsample_bytree=0.7,
)
model.fit(X_train, y_train)
```

### 4️⃣ Model Evaluation
- Cross-validation (10 folds)
- Confusion matrix
- Precision, Recall, F1-score

---

### 5️⃣ Model Persistence
- The trained model is saved for deployment using pickle:
```python
import pickle

with open('model.pkl', 'wb') as f:
    pickle.dump(model, f)
```

### Web Application
The Streamlit app allows users to input health parameters and get real-time heart disease predictions.

## Features:
- Interactive sliders and dropdowns
- Predict button to compute results
- Result display: "No Heart Disease" or "High Probability of Heart Disease"
- Sample Prediction Function:

```python
def prediction(Age, Sex, ChestPainType, RestingBP, Cholesterol, FastingBS, RestingECG, MaxHR, ExerciseAngina, Oldpeak, ST_Slope):
    df = pd.DataFrame(columns=Inputs)
    df.at[0, "Age"] = Age
    df.at[0, "RestingBP"] = RestingBP
    df.at[0, "Cholesterol"] = Cholesterol
    df.at[0, "MaxHR"] = MaxHR
    df.at[0, "Oldpeak"] = Oldpeak
    df.at[0, "FastingBS"] = 1 if FastingBS == "Yes" else 0
    df.at[0, "Sex"] = "M" if Sex == "Male" else "F"
    df.at[0, "ExerciseAngina"] = "Y" if ExerciseAngina == "Yes" else "N"

    df_encoded = pd.get_dummies(df, columns=['Sex', 'ChestPainType', 'RestingECG', 'ExerciseAngina', 'ST_Slope'], drop_first=True) * 1
    result = Model.predict(df_encoded)[0]

    return result
```
## Results  
✅ **Cross-Validation Score**: **88%**  
✅ **Train Accuracy**: **90%**  
✅ **Test Accuracy**: **94%**  

### 📊 Model Performance Metrics  
| Metric      | Score  |
|------------|--------|
| Precision  | 94%    |
| Recall     | 94%    |
| F1-Score   | 94%    |

The model was evaluated using **cross-validation**, confusion matrix, and classification metrics such as **precision, recall, and F1-score**.  

---
## Connect with Me  

💻 **GitHub**: [Ahmed-Ramadan-Ahmed](https://github.com/Ahmed-Ramadan-Ahmed)  
📇 **LinkedIn**: [Ahmed Ramadan](https://www.linkedin.com/in/ahmed-ramadan-348264225/)  
🎯 **LeetCode**: [A_Ramadan_A](https://leetcode.com/u/A_Ramadan_A/)  
🏆 **Codeforces**: [A.R.A](https://codeforces.com/profile/A.R.A)  

---
