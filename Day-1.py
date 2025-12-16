# STEP 1: Load dataset
from sklearn.datasets import load_breast_cancer
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score,
    f1_score, confusion_matrix, classification_report
)

dataset = load_breast_cancer()

# STEP 2: Convert dataset to DataFrame
df = pd.DataFrame(dataset.data, columns=dataset.feature_names)
df['target'] = dataset.target

# Reduce features to control accuracy
X = df.iloc[:, :10]
y = df['target']

# Larger test size
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.4, random_state=42
)

# STEP 3: Logistic Regression with regularization
model = LogisticRegression(C=0.1, max_iter=10000)
model.fit(X_train, y_train)

# Predictions
y_pred = model.predict(X_test)

# STEP 4: Model Performance
accuracy = accuracy_score(y_test, y_pred)
precision = precision_score(y_test, y_pred)
recall = recall_score(y_test, y_pred)
f1 = f1_score(y_test, y_pred)
cm = confusion_matrix(y_test, y_pred)

print("Model Performance Metrics")
print("-------------------------")
print("Accuracy :", accuracy)
print("Precision:", precision)
print("Recall   :", recall)
print("F1 Score :", f1)

print("\nConfusion Matrix:")
print(cm)

print("\nClassification Report:\n")
print(classification_report(y_test, y_pred))
