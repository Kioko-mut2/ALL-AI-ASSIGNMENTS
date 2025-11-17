# This script mirrors notebook steps; prefer to use as notebook cells.

import pandas as pd
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, f1_score, classification_report
import joblib
import numpy as np

# Load dataset (replace with pd.read_csv("path/to/kaggle.csv") if available)
data = load_breast_cancer(as_frame=True)
df = data.frame

# Create synthetic 'priority' label from a numeric feature (e.g., mean radius)
# Map low/medium/high from quantiles of 'mean radius' for demo purposes
q = np.quantile(df['mean radius'], [0.33, 0.66])
def to_priority(x):
    if x <= q[0]:
        return "low"
    elif x <= q[1]:
        return "medium"
    else:
        return "high"
df['priority'] = df['mean radius'].apply(to_priority)

# Features and label
X = df.drop(columns=['target', 'priority'])
y = df['priority']

# Preprocess
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
scaler = StandardScaler()
X_train_s = scaler.fit_transform(X_train)
X_test_s = scaler.transform(X_test)

# Train Random Forest
clf = RandomForestClassifier(n_estimators=200, random_state=42, class_weight='balanced')
clf.fit(X_train_s, y_train)

# Evaluate
y_pred = clf.predict(X_test_s)
acc = accuracy_score(y_test, y_pred)
f1_macro = f1_score(y_test, y_pred, average='macro')
report = classification_report(y_test, y_pred)

print("Accuracy:", acc)
print("F1 (macro):", f1_macro)
print(report)

# Save model and scaler
joblib.dump(clf, "rf_priority_model.joblib")
joblib.dump(scaler, "scaler.joblib")