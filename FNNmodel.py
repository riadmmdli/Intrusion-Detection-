import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
import pandas as pd
import numpy as np
from sklearn.impute import SimpleImputer

# Check if GPU is available
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Load dataset
df = pd.read_csv('C:/Users/riadm/Desktop/DNP3 Intrusion Detection/DNP3_Merged_Dataset.csv', low_memory=False)

# Strip whitespace from column names
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

# Encode the target label
le = LabelEncoder()
df['Label'] = le.fit_transform(df['Label'])

# Drop non-relevant columns
df.drop(['Unnamed: 0', 'flow ID', 'source IP', 'destination IP',
         'source port', 'destination port', 'protocol', 'date'], axis=1, inplace=True, errors='ignore')

# Separate features and target
X = df.drop('Label', axis=1)
y = df['Label']

# Select only numeric features
X = X.select_dtypes(include=[np.number])

# Feature scaling
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Split into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)

# Convert to PyTorch tensors and move to GPU (if available)
X_train_tensor = torch.tensor(X_train, dtype=torch.float32).to(device)
X_test_tensor = torch.tensor(X_test, dtype=torch.float32).to(device)
y_train_tensor = torch.tensor(y_train.values, dtype=torch.long).to(device)
y_test_tensor = torch.tensor(y_test.values, dtype=torch.long).to(device)

# Define the neural network architecture
class SimpleNN(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim):
        super(SimpleNN, self).__init__()
        self.fc1 = nn.Linear(input_dim, hidden_dim)  # Fully connected layer
        self.fc2 = nn.Linear(hidden_dim, output_dim)  # Output layer
        self.relu = nn.ReLU()  # ReLU activation function
        self.softmax = nn.Softmax(dim=1)  # For multi-class classification

    def forward(self, x):
        x = self.relu(self.fc1(x))  # Apply ReLU activation to hidden layer
        x = self.fc2(x)  # Output layer (no activation here)
        return self.softmax(x)  # Softmax for multi-class output

# Initialize the model, loss function, and optimizer
input_dim = X_train.shape[1]  # Number of features
hidden_dim = 64  # You can experiment with this value
output_dim = len(np.unique(y_train))  # Number of classes

model = SimpleNN(input_dim, hidden_dim, output_dim).to(device)  # Move the model to GPU if available
criterion = nn.CrossEntropyLoss()  # Suitable for classification tasks
optimizer = optim.Adam(model.parameters(), lr=0.001)  # Adam optimizer

# Training the model
num_epochs = 10
batch_size = 64
train_data = TensorDataset(X_train_tensor, y_train_tensor)
train_loader = DataLoader(train_data, batch_size=batch_size, shuffle=True)

for epoch in range(num_epochs):
    model.train()  # Set the model to training mode
    running_loss = 0.0
    for batch in train_loader:
        inputs, labels = batch
        inputs, labels = inputs.to(device), labels.to(device)  # Move to GPU if available

        # Zero the gradients
        optimizer.zero_grad()

        # Forward pass: compute predicted outputs by passing inputs to the model
        outputs = model(inputs)

        # Compute the loss
        loss = criterion(outputs, labels)

        # Backward pass: compute gradient of the loss with respect to the model's parameters
        loss.backward()

        # Update the model parameters
        optimizer.step()

        # Update the running loss
        running_loss += loss.item()

    # Print statistics every epoch
    print(f'Epoch [{epoch+1}/{num_epochs}], Loss: {running_loss/len(train_loader):.4f}')

# Evaluation on test set
model.eval()  # Set the model to evaluation mode
with torch.no_grad():
    # Make predictions
    outputs = model(X_test_tensor)
    _, predicted = torch.max(outputs, 1)  # Get the index of the max probability (predicted class)

# Evaluate performance (e.g., accuracy)
accuracy = (predicted.cpu() == y_test_tensor.cpu()).float().mean()
print(f'Accuracy on Test Data: {accuracy * 100:.2f}%')

# Confusion Matrix
from sklearn.metrics import confusion_matrix
import seaborn as sns
import matplotlib.pyplot as plt

cm = confusion_matrix(y_test_tensor.cpu(), predicted.cpu())
plt.figure(figsize=(6, 5))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=le.classes_, yticklabels=le.classes_)
plt.title('Confusion Matrix')
plt.xlabel('Predicted')
plt.ylabel('Actual')
plt.show()

# Classification Report
from sklearn.metrics import classification_report
print("Classification Report:")
print(classification_report(y_test_tensor.cpu(), predicted.cpu(), target_names=le.classes_))

# ROC-AUC Score (if binary classification)
from sklearn.metrics import roc_auc_score
y_prob = model(X_test_tensor).cpu().detach().numpy()[:, 1]  # Probabilities for positive class
roc_auc = roc_auc_score(y_test_tensor.cpu(), y_prob)
print(f'ROC-AUC Score: {roc_auc:.2f}')
