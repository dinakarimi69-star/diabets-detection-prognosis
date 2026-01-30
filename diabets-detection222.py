# =========================
# 1. Import Libraries
# =========================
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, confusion_matrix

# =========================
# 2. Load Dataset
# =========================
df = pd.read_csv("diabets-datasets.csv")

# Check column names to verify correct loading
print("Dataset Loaded Successfully")
print("Columns in the dataset:", df.columns)

# =========================
# 3. Data Preprocessing (Encoding Categorical Data)
# =========================
# Remove any extra spaces in column names
df.columns = df.columns.str.strip()

# Check columns again after stripping spaces
print("\nUpdated Columns:", df.columns)

# Columns to encode
columns_to_encode = [
    "Fatigue", 
    "Excess_Thirst", 
    "Frequent_Urination", 
    "Blurred_Vision", 
    "Weight_Loss", 
    "Diabetes_Status"
]

# Initialize LabelEncoder
encoder = LabelEncoder()

# Check and encode columns
for col in columns_to_encode:
    if col in df.columns:
        df[col] = encoder.fit_transform(df[col].astype(str))  # Ensure everything is converted to string before encoding
    else:
        print(f"Warning: Column '{col}' not found in the dataset!")

# =========================
# 4. Check for Missing Values
# =========================
print("\nMissing values in dataset:")
print(df.isnull().sum())

# Fill missing values with mode (for categorical columns)
df = df.fillna(df.mode().iloc[0])

# =========================
# 5. Convert All Columns to Numeric
# =========================
# Convert all non-numeric columns (if any) to numeric using LabelEncoder
for col in df.columns:
    if df[col].dtype == 'object':  # Check if the column is still of type 'object'
        df[col] = encoder.fit_transform(df[col].astype(str))

# Check the data types of columns
print("\nData types after encoding:")
print(df.dtypes)

# =========================
# 6. Split Data into Features (X) and Target (y)
# =========================
X = df.drop(["ID", "Diabetes_Status"], axis=1)  # Features (remove 'ID' and 'Diabetes_Status' column)
y = df["Diabetes_Status"]  # Target (Diabetes_Status column)

# Split the dataset into training and test sets (70% train, 30% test)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# =========================
# 7. Train Model
# =========================
# Using Random Forest Classifier for training
model = RandomForestClassifier(random_state=42)
model.fit(X_train, y_train)

# =========================
# 8. Make Predictions and Evaluate Accuracy
# =========================
y_pred = model.predict(X_test)

# Calculate accuracy
accuracy = accuracy_score(y_test, y_pred)
print(f"\nModel Accuracy: {accuracy * 100:.2f}%")

# =========================
# 9. Confusion Matrix (Matplotlib Debugging)
# =========================
cm = confusion_matrix(y_test, y_pred)
print("\nConfusion Matrix:")
print(cm)

# Plotting Confusion Matrix
plt.figure(figsize=(6, 6))
plt.imshow(cm, cmap='Blues', alpha=0.7)
plt.title("Confusion Matrix")
plt.xlabel("Predicted Labels")
plt.ylabel("True Labels")

# Add text annotations to the matrix
for i in range(cm.shape[0]):
    for j in range(cm.shape[1]):
        plt.text(j, i, cm[i, j], ha='center', va='center', color='red')

plt.colorbar()
plt.show()

# =========================
# 10. Visualize Feature Importance
# =========================
# Get the importance of each feature
importances = model.feature_importances_
features = X.columns

# Plotting Feature Importances
plt.figure(figsize=(10, 6))
plt.barh(features, importances, color='skyblue')
plt.xlabel("Importance Score")
plt.title("Feature Importances for Diabetes Detection")
plt.show()
