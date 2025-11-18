# Hospital Patient Readmission Prediction Notebook
# ================================================
# Author: [Your Name]
# Date: [Insert Date]
# Description: Predicting patient readmission within 30 days using AI.

# Import libraries
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import confusion_matrix, precision_score, recall_score
import joblib

# -----------------------------------------
# 1. Simulate Hypothetical Patient Dataset
# -----------------------------------------
# For demonstration, we'll create 200 patients with features
np.random.seed(42)
data = pd.DataFrame({
    'age': np.random.randint(20, 90, 200),
    'gender': np.random.choice(['Male', 'Female'], 200),
    'num_previous_admissions': np.random.randint(0, 5, 200),
    'length_of_stay': np.random.randint(1, 20, 200),
    'comorbidity_score': np.random.randint(0, 5, 200),
    'discharge_type': np.random.choice(['Home', 'Rehab', 'Nursing'], 200),
    'readmitted_30days': np.random.choice([0,1], 200, p=[0.7,0.3])
})
print("Sample Data:\n", data.head())

# -----------------------------------------
# 2. Preprocessing
# -----------------------------------------
# Separate features and target
X = data.drop('readmitted_30days', axis=1)
y = data['readmitted_30days']

# Identify categorical and numerical columns
cat_features = ['gender', 'discharge_type']
num_features = ['age', 'num_previous_admissions', 'length_of_stay', 'comorbidity_score']

# Preprocessing pipeline
preprocessor = ColumnTransformer(transformers=[
    ('num', StandardScaler(), num_features),
    ('cat', OneHotEncoder(drop='first'), cat_features)
])

X_processed = preprocessor.fit_transform(X)

# -----------------------------------------
# 3. Train/Test Split
# -----------------------------------------
X_train, X_test, y_train, y_test = train_test_split(X_processed, y, test_size=0.2, random_state=42)

# -----------------------------------------
# 4. Model Training
# -----------------------------------------
rf_model = RandomForestClassifier(n_estimators=100, max_depth=5, random_state=42)
rf_model.fit(X_train, y_train)

# -----------------------------------------
# 5. Evaluation
# -----------------------------------------
y_pred = rf_model.predict(X_test)
cm = confusion_matrix(y_test, y_pred)
precision = precision_score(y_test, y_pred)
recall = recall_score(y_test, y_pred)

print("Confusion Matrix:\n", cm)
print("Precision:", precision)
print("Recall:", recall)

# -----------------------------------------
# 6. Deployment Simulation
# -----------------------------------------
# Save the trained model
joblib.dump(rf_model, 'readmission_model.pkl')
# Save the preprocessor
joblib.dump(preprocessor, 'preprocessor.pkl')

# Load model and make a prediction for a new patient
loaded_model = joblib.load('readmission_model.pkl')
loaded_preprocessor = joblib.load('preprocessor.pkl')

# Example new patient
new_patient = pd.DataFrame({
    'age':[65],
    'gender':['Female'],
    'num_previous_admissions':[2],
    'length_of_stay':[10],
    'comorbidity_score':[3],
    'discharge_type':['Home']
})
new_patient_processed = loaded_preprocessor.transform(new_patient)
prediction = loaded_model.predict(new_patient_processed)
print("Readmission Prediction (0=No, 1=Yes):", prediction[0])
