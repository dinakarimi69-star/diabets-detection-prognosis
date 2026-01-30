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
df = pd.read_csv("diabets_datasets.csv")

print("Dataset Loaded Successfully")
print(df.head())


# =========================
# 3. Encode Categorical Data
# =========================
encoder = LabelEncoder()

columns_to_encode = [
    "Fatigue",
    "Excess_Thirst",
    "Frequent_Urination",
    "Blurred_Vision",
    "Weight_Loss",
    "Diabetes_Status"
]

for col in columns_to_encode:
    df[col] = encoder.fit_transform(df[col])


# =========================
# 4. Split Features & Target
# =========================
X = df.drop(["ID", "Diabetes_Status"], axis=1)
y = df["Diabetes_Status"]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.3, random_state=42
)


# =========================
# 5. Train Model
# =========================
model = RandomForestClassifier(random_state=42)
model.fit(X_train, y_train)


# =========================
# 6. Prediction & Accuracy
# =========================
y_pred = model.predict(X_test)

accuracy = accuracy_score(y_test, y_pred)
print(f"\nModel Accuracy: {accuracy * 100:.2f}%")


# =========================
# 7. Confusion Matrix (Matplotlib Debugging)
# =========================
cm = confusion_matrix(y_test, y_pred)
print("\nConfusion Matrix:\n", cm)

plt.figure()
plt.imshow(cm)
plt.title("Confusion Matrix")
plt.xlabel("Predicted Label")
plt.ylabel("True Label")

for i in range(cm.shape[0]):
    for j in range(cm.shape[1]):
        plt.text(j, i, cm[i, j], ha="center", va="center")

plt.show()


# =========================
# 8. Feature Importance Visualization
# =========================
importances = model.feature_importances_
features = X.columns

plt.figure()
plt.barh(features, importances)
plt.xlabel("Importance Score")
plt.title("Feature Importance for Diabetes Detection")
plt.show()


