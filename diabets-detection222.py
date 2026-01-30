
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, confusion_matrix

df = pd.read_csv("diabets-datasets.csv")

print("Dataset Loaded Successfully")
print("Columns in the dataset:", df.columns)

df.columns = df.columns.str.strip()


print("\nUpdated Columns:", df.columns)


columns_to_encode = [
    "Fatigue", 
    "Excess_Thirst", 
    "Frequent_Urination", 
    "Blurred_Vision", 
    "Weight_Loss", 
    "Diabetes_Status"
]


encoder = LabelEncoder()

for col in columns_to_encode:
    if col in df.columns:
        df[col] = encoder.fit_transform(df[col].astype(str))  # Ensure everything is converted to string before encoding
    else:
        print(f"Warning: Column '{col}' not found in the dataset!")

print("\nMissing values in dataset:")
print(df.isnull().sum())


df = df.fillna(df.mode().iloc[0])

for col in df.columns:
    if df[col].dtype == 'object':  # Check if the column is still of type 'object'
        df[col] = encoder.fit_transform(df[col].astype(str))


print("\nData types after encoding:")
print(df.dtypes)

X = df.drop(["ID", "Diabetes_Status"], axis=1)  # Features (remove 'ID' and 'Diabetes_Status' column)
y = df["Diabetes_Status"]  # Target (Diabetes_Status column)


X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)


model = RandomForestClassifier(random_state=42)
model.fit(X_train, y_train)

y_pred = model.predict(X_test)

accuracy = accuracy_score(y_test, y_pred)
print(f"\nModel Accuracy: {accuracy * 100:.2f}%")


cm = confusion_matrix(y_test, y_pred)
print("\nConfusion Matrix:")
print(cm)


plt.figure(figsize=(6, 6))
plt.imshow(cm, cmap='Blues', alpha=0.7)
plt.title("Confusion Matrix")
plt.xlabel("Predicted Labels")
plt.ylabel("True Labels")


for i in range(cm.shape[0]):
    for j in range(cm.shape[1]):
        plt.text(j, i, cm[i, j], ha='center', va='center', color='red')

plt.colorbar()
plt.show()


importances = model.feature_importances_
features = X.columns

plt.figure(figsize=(10, 6))
plt.barh(features, importances, color='skyblue')
plt.xlabel("Importance Score")
plt.title("Feature Importances for Diabetes Detection")
plt.show()

