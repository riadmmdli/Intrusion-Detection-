# 🔐 DNP3 Intrusion Detection System

This project implements a machine learning-based Intrusion Detection System (IDS) targeting the DNP3 protocol — a critical communication standard in industrial control systems (ICS) and SCADA networks. The system detects various attack types using classical machine learning models after a comprehensive data preprocessing pipeline.

---

## 📁 Dataset Description

The dataset used in this project is the **CIC-DNP3 2023 Intrusion Detection Dataset**, which includes both normal and malicious traffic. It simulates real-world DNP3 attacks and includes feature-rich CSV logs extracted from packet captures.

### 📂 Attack Types Included:
- Disable Unsolicited Messages
- Cold Restart
- Warm Restart
- Enumerate
- Info Leak
- Initialize Data
- MITM DoS
- Replay Attack
- Stop Application

🔗 Dataset Link: https://zenodo.org/records/7348493/files/DNP3_Intrusion_Detection_Dataset_Final.7z?download=1

---

## ⚙️ Project Structure
```
📦 DNP3-Intrusion-Detection
├── .gitignore # Ignores the final merged dataset
├── DNP3_Merged_Dataset.csv # Final processed dataset (ignored from Git)
├── DataPrepocessing.py # Script to merge and label raw dataset folders
├── model_train2.py # Full ML pipeline: preprocessing, training, evaluation
├── README.md # Project documentation
```
---

## 🧪 Machine Learning Models

We trained and evaluated the following models:

| Model                | Description                                        |
|---------------------|----------------------------------------------------|
| ✅ Random Forest      | Ensemble of decision trees (fast + accurate)       |
| ✅ XGBoost            | Gradient Boosted Trees (optimized for tabular data)|
| ✅ Logistic Regression| Linear baseline model                             |

---

## 🧼 Data Preprocessing

All steps were automated inside `model_train2.py`:

1. **Drop features** with over 95% missing values
2. **Imputation**:
   - Mean for numeric columns
   - Mode for categorical columns
3. **Encoding**: Labels are encoded using `LabelEncoder`
4. **Drop irrelevant features**:
   - `flow ID`, `source IP`, `destination port`, etc.
5. **Scaling**: `StandardScaler` used on all features
6. **Train/Test Split**: Stratified 80% train / 20% test

---

## 📊 Evaluation Metrics

Each model is evaluated using:

- ✅ Accuracy
- ✅ Classification Report (Precision, Recall, F1-score)
- ✅ Confusion Matrix (Seaborn heatmap)
- ✅ Precision-Recall Curve
- ✅ Accuracy comparison bar plot

---

## ✅ Results Summary

| Model                | Accuracy | Macro F1-Score | Weighted F1-Score |
|---------------------|----------|----------------|-------------------|
| Random Forest        | 0.8889   | 0.7142         | 0.8887            |
| XGBoost              | 0.8375   | 0.6677         | 0.8373            |
| Logistic Regression  | 0.7668   | 0.5664         | 0.7622            |

📌 **Observations**:
- **Random Forest** achieved the highest overall accuracy and weighted F1-score, showing superior performance across most classes.
- **XGBoost** performed competitively but was slightly behind in both metrics.
- **Logistic Regression** struggled on minority classes like `MITM_DOS` and `No Label`, resulting in lower macro F1-scores.

🎯 These scores highlight the effectiveness of ensemble models like Random Forest for multi-class intrusion detection tasks.

Note: Bar plots and confusion matrices are displayed during training automatically.

---

## 🚀 How to Run

### Step 1: Merge and Label Dataset
Ensure the raw CSV folders are structured correctly in the dataset root path, then run:

```bash
python DataPrepocessing.py
```
This script reads and merges all labeled CSVs into DNP3_Merged_Dataset.csv.

Step 2: Train and Evaluate Models
```bash
python model_train2.py
```
This script handles:

- Feature selection

- Label encoding

- Data splitting

- Training 3 classifiers

- Plotting evaluation metrics

👥 Contributors
Emel Tuğçe Kara – 211015010

Riad Memmedli – 211015082




