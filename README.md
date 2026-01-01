# Human Activity Recognition using Machine Learning (WISDM Dataset)
This project implements an **end-to-end Human Activity Recognition (HAR)** system using Machine Learning on accelerometer sensor data from the **WISDM dataset**.   The trained model is deployed using **Streamlit** to enable real-time activity prediction.

---

## Project Overview

Human Activity Recognition plays a crucial role in applications such as:
- Fitness tracking
- Healthcare monitoring
- Smart devices
- Context-aware systems

This project classifies human activities based on **X, Y, Z accelerometer readings** using supervised machine learning models.

---

## Key Features

- End-to-end ML pipeline (data loading → training → deployment)
- Uses raw **WISDM accelerometer dataset (.txt)**
- Missing value handling
- Activity label encoding
- Feature scaling using **StandardScaler**
- Trained and compared multiple ML models:
  - Random Forest Classifier
  - XGBoost Classifier
- Model evaluation using:
  - Accuracy
  - Weighted F1-score
  - Confusion Matrix
  - Classification Report
- Best model selection based on **F1-score**
- Model persistence using **Joblib**
- Interactive **Streamlit Web Application**

---

## Machine Learning Workflow

1. Load raw WISDM dataset
2. Assign meaningful column names
3. Remove missing values
4. Encode activity labels
5. Select accelerometer features (X, Y, Z)
6. Train-test split (80% / 20%)
7. Build preprocessing + model pipelines
8. Train Random Forest and XGBoost models
9. Evaluate models using F1-score and accuracy
10. Select the best-performing model
11. Save trained model and label encoder
12. Deploy using Streamlit for real-time prediction

---

## Dataset Information

- **Dataset**: WISDM (Wireless Sensor Data Mining)
- **File format**: `.txt`
- **Features**:
  - X-acceleration
  - Y-acceleration
  - Z-acceleration
- **Target**: Activity label  
  Examples:
  - Walking
  - Jogging
  - Sitting
  - Standing
  - Upstairs
  - Downstairs

---

## Tech Stack

**Language**
- Python

**Libraries**
- Pandas
- NumPy
- Scikit-learn
- XGBoost
- Matplotlib
- Seaborn
- Joblib
- Streamlit

---

## Model Evaluation Metrics

The models are evaluated using:

- **Accuracy** – Overall correctness
- **F1 Score (Weighted)** – Suitable for multi-class classification
- **Confusion Matrix** – Class-wise prediction analysis
- **Classification Report** – Precision, Recall, F1-score

 The **best model is selected based on the weighted F1-score**.

---

<img width="1919" height="913" alt="image" src="https://github.com/user-attachments/assets/489fc22d-2ac4-495a-9baf-ce82e741dc85" />

## App link

https://lxp2knweygfnfirmvfrnt5.streamlit.app/

## Model Saving

```python
import joblib
joblib.dump(best_model, "wisdm_model.pkl")
joblib.dump(label_encoder, "label_encoder.pkl")
```

## Streamlit Web Application

The Streamlit app allows users to:
  - Enter real-time accelerometer sensor readings
  - Predict human activity instantly
  - View predicted activity class for real-time decision support

Input Categories: Sensor Data
  
  Accelerometer Readings
    - X Acceleration
    - Y Acceleration
    - Z Acceleration
  These values represent real-time motion data collected from wearable or mobile sensors.

Model Inference

  - The trained Machine Learning model processes the input accelerometer values
  - Feature scaling is applied automatically using the trained preprocessing pipeline
  - The model predicts the most probable human activity class

## Results

Displays:

  - Best Model Name
  - Best Weighted F1 Score
  - Model comparison summary

Prediction Output:

  Predicted Activity Name 
    - Walking
    - Jogging
    - Sitting
    - Standing
    - Upstairs
    - Downstairs

## Author

Dhilip K
