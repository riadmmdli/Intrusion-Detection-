import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import tqdm
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score, confusion_matrix, classification_report,
    roc_auc_score, roc_curve, auc, precision_recall_curve
)
from sklearn.impute import SimpleImputer
import time

# Load dataset
df = pd.read_csv('C:/Users/riadm/Desktop/DNP3 Intrusion Detection/DNP3_Merged_Dataset.csv', low_memory=False)

# Clean column names
df.columns = df.columns.str.strip()

# Drop columns with over 95% missing data
missing_ratios = df.isnull().mean()
df = df.loc[:, missing_ratios < 0.95]

# Drop rows missing target
df = df.dropna(subset=['Label'])

# Impute numeric features
numerical_cols = df.select_dtypes(include=[np.number]).columns
imputer = SimpleImputer(strategy='mean')
df[numerical_cols] = imputer.fit_transform(df[numerical_cols])

# Impute categorical features
categorical_cols = df.select_dtypes(include=['object']).columns
for col in categorical_cols:
    df[col] = df[col].fillna(df[col].mode()[0])

# Encode target
le = LabelEncoder()
df['Label'] = le.fit_transform(df['Label'])

# Drop irrelevant columns
df.drop(['Unnamed: 0', 'flow ID', 'source IP', 'destination IP',
         'source port', 'destination port', 'protocol', 'date'], axis=1, inplace=True, errors='ignore')

# Features and labels
X = df.drop('Label', axis=1).select_dtypes(include=[np.number])
y = df['Label']

# Scale features
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)

# Models
models = {
    'Random Forest': RandomForestClassifier(n_estimators=100, random_state=42),
    'SVM': SVC(random_state=42),  # Removed probability=True for speed
    'Logistic Regression': LogisticRegression(max_iter=1000, random_state=42)
}

# Train models
trained_models = {}
for name in tqdm(models, desc="Training Models"):
    start_time = time.time()
    model = models[name]
    model.fit(X_train, y_train)
    end_time = time.time()
    print(f"{name} training completed in {end_time - start_time:.2f} seconds.")
    trained_models[name] = model

# Predictions
y_preds = {name: model.predict(X_test) for name, model in trained_models.items()}

# Accuracy comparison
accuracies = [accuracy_score(y_test, y_preds[name]) for name in models]

plt.figure(figsize=(8, 6))
sns.barplot(x=list(models.keys()), y=accuracies, palette='viridis')
plt.title('Model Accuracy Comparison')
plt.xlabel('Models')
plt.ylabel('Accuracy')
plt.ylim(0, 1)
plt.show()

# Confusion matrix
def plot_confusion_matrix(y_true, y_pred, model_name):
    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(6, 5))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=['Normal', 'Attack'], yticklabels=['Normal', 'Attack'])
    plt.title(f'Confusion Matrix - {model_name}')
    plt.xlabel('Predicted')
    plt.ylabel('Actual')
    plt.show()

for name in models:
    plot_confusion_matrix(y_test, y_preds[name], name)

# ROC Curve
def plot_roc_curve(y_true, y_prob, model_name):
    fpr, tpr, _ = roc_curve(y_true, y_prob)
    roc_auc = auc(fpr, tpr)
    plt.figure(figsize=(8, 6))
    plt.plot(fpr, tpr, label=f'{model_name} (AUC = {roc_auc:.2f})', lw=2)
    plt.plot([0, 1], [0, 1], 'k--', lw=1)
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title(f'ROC Curve - {model_name}')
    plt.legend(loc='lower right')
    plt.grid(True)
    plt.show()

# PR Curve
def plot_precision_recall_curve(y_true, y_prob, model_name):
    precision, recall, _ = precision_recall_curve(y_true, y_prob)
    plt.figure(figsize=(8, 6))
    plt.plot(recall, precision, color='darkgreen', lw=2)
    plt.xlabel('Recall')
    plt.ylabel('Precision')
    plt.title(f'Precision-Recall Curve - {model_name}')
    plt.grid(True)
    plt.show()

# Plot ROC and PR curves only for models that support predict_proba
for name, model in trained_models.items():
    if hasattr(model, "predict_proba"):
        y_prob = model.predict_proba(X_test)[:, 1]
        plot_roc_curve(y_test, y_prob, name)
        plot_precision_recall_curve(y_test, y_prob, name)

# Evaluation Summary
def evaluate_model(model_name, model, y_pred):
    print(f"\nModel: {model_name}")
    print(f"Accuracy: {accuracy_score(y_test, y_pred) * 100:.2f}%")
    print("Confusion Matrix:")
    print(confusion_matrix(y_test, y_pred))
    print("Classification Report:")
    print(classification_report(y_test, y_pred))
    if hasattr(model, "predict_proba"):
        print(f"ROC-AUC: {roc_auc_score(y_test, model.predict_proba(X_test)[:, 1]):.2f}")

for name, model in trained_models.items():
    evaluate_model(name, model, y_preds[name])
