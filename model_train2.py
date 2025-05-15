import pandas as pd
import numpy as np
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score, accuracy_score
import seaborn as sns
import matplotlib.pyplot as plt
from tqdm import tqdm
from xgboost import XGBClassifier

# Load dataset
df = pd.read_csv('C:/Users/riadm/Desktop/DNP3 Intrusion Detection/DNP3_Merged_Dataset.csv', low_memory=False)
df.columns = df.columns.str.strip()

# Drop columns with more than 95% missing values
missing_ratios = df.isnull().mean()
df = df.loc[:, missing_ratios < 0.95]

# Drop rows with missing target label
df = df.dropna(subset=['Label'])

# Impute numerical columns with mean
numerical_cols = df.select_dtypes(include=[np.number]).columns
imputer = SimpleImputer(strategy='mean')
df[numerical_cols] = imputer.fit_transform(df[numerical_cols])

# Impute categorical columns with mode
categorical_cols = df.select_dtypes(include=['object']).columns
for col in categorical_cols:
    df[col] = df[col].fillna(df[col].mode()[0])

# Encode target label
le = LabelEncoder()
df['Label'] = le.fit_transform(df['Label'])

# Drop non-relevant columns
df.drop(['Unnamed: 0', 'flow ID', 'source IP', 'destination IP',
         'source port', 'destination port', 'protocol', 'date'], axis=1, inplace=True, errors='ignore')

# Features and target
X = df.drop('Label', axis=1).select_dtypes(include=[np.number])
y = df['Label']

# Standard scaling
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Stratified train-test split
X_train, X_test, y_train, y_test = train_test_split(
    X_scaled, y, test_size=0.2, stratify=y, random_state=42)

# Evaluation function
def evaluate_model(y_test, y_pred, y_prob, model_name):
    eval_results = {}
    eval_results['model_name'] = model_name
    eval_results['classification_report'] = classification_report(
        y_test, y_pred, target_names=le.classes_, output_dict=True, zero_division=0)

    cm = confusion_matrix(y_test, y_pred)
    eval_results['confusion_matrix'] = cm

    eval_results['accuracy'] = accuracy_score(y_test, y_pred)

    if len(np.unique(y_test)) == 2:
        roc_auc = roc_auc_score(y_test, y_prob[:, 1])
        eval_results['roc_auc'] = roc_auc

    return eval_results

# Progress bar wrapper for training and prediction
def train_and_evaluate_model(model, name):
    print(f"\nTraining {name}...")
    for _ in tqdm(range(1), desc=f"Fitting {name}"):
        model.fit(X_train, y_train)

    for _ in tqdm(range(1), desc=f"Predicting with {name}"):

        y_pred = model.predict(X_test)
        y_prob = model.predict_proba(X_test)

    # Display test set class distribution
    print("\nTest Set Label Distribution:")
    print(pd.Series(y_test).value_counts().sort_index())
    print("\nPredicted Label Distribution:")
    print(pd.Series(y_pred).value_counts().sort_index())

    eval_results = evaluate_model(y_test, y_pred, y_prob, name)
    return eval_results

# Store evaluation results for all models
results = []

# Random Forest
rf_model = RandomForestClassifier(n_estimators=100, random_state=42)
results.append(train_and_evaluate_model(rf_model, "Random Forest"))

# XGBoost (CPU)
xgb_model = XGBClassifier(
    n_estimators=100,
    max_depth=6,
    learning_rate=0.1,
    subsample=0.8,
    colsample_bytree=0.8,
    use_label_encoder=False,
    eval_metric='mlogloss',
    tree_method='hist',  # CPU-optimized
    random_state=42
)
results.append(train_and_evaluate_model(xgb_model, "XGBoost"))

# Logistic Regression with class balancing
logreg_model = LogisticRegression(max_iter=1000, class_weight='balanced', random_state=42)
results.append(train_and_evaluate_model(logreg_model, "Logistic Regression"))

# Print all evaluation results at the end
for result in results:
    print(f"\n--- {result['model_name']} ---")
    print("Classification Report:")
    report_df = pd.DataFrame(result['classification_report']).transpose()
    print(report_df)

    # Also print overall accuracy for clarity
    print(f"\nAccuracy: {result['accuracy']:.4f}")

    # Plot confusion matrix
    cm = result['confusion_matrix']
    plt.figure(figsize=(6, 5))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=le.classes_, yticklabels=le.classes_)
    plt.title(f'{result["model_name"]} - Confusion Matrix')
    plt.xlabel('Predicted')
    plt.ylabel('Actual')
    plt.tight_layout()
    plt.show()

    if 'roc_auc' in result:
        print(f'{result["model_name"]} ROC-AUC Score: {result["roc_auc"]:.2f}')
