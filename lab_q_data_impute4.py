# Data 404 AI-Based Medical Diagnostic Support Framework Using Public Health Data
# Group Project - Hybrid Imputation with Pipeline (Research Best Practice)
# This approach:
# Splits the data before imputation to avoid "data leakage" (https://scikit-learn.org/stable/common_pitfalls.html)


import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.experimental import enable_iterative_imputer  # noqa: F401
from sklearn.impute import IterativeImputer, SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.svm import SVC
from sklearn.metrics import classification_report, accuracy_score
import joblib

# Load original data WITH missing values
df = pd.read_excel('lab_q_data.xlsx')

# Remove class 9 (only 1 sample)
df = df[df['Diabetes'] != 9]

# Separate features and target
X = df.drop(['SEQN', 'Diabetes'], axis=1)
y = df['Diabetes']

# Split FIRST (prevents data leakage)
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# Show class distribution
print()
print("CLASS DISTRIBUTION")
print("=" * 60)

# Create a table for Full Dataset (Google)
print("\nFull Dataset:")
print("-" * 40)
print(f"{'Class':<15} {'Count':<10} {'Percentage'}")
print("-" * 40)
for idx in sorted(y.unique()):
    count = (y == idx).sum()
    pct = (count / len(y)) * 100
    label = {1: 'Diabetes', 2: 'No Diabetes', 3: 'Borderline'}[idx]
    print(f"{idx} ({label:<11}) {count:<10} {pct:.2f}%")
print("-" * 40)
print(f"{'Total':<15} {len(y):<10} 100.00%")

# Training Set
print("\n\nTraining Set:")
print("-" * 40)
print(f"{'Class':<15} {'Count':<10} {'Percentage'}")
print("-" * 40)
for idx in sorted(y_train.unique()):
    count = (y_train == idx).sum()
    pct = (count / len(y_train)) * 100
    label = {1: 'Diabetes', 2: 'No Diabetes', 3: 'Borderline'}[idx]
    print(f"{idx} ({label:<11}) {count:<10} {pct:.2f}%")
print("-" * 40)
print(f"{'Total':<15} {len(y_train):<10} 100.00%")

# Test Set
print("\n\nTest Set:")
print("-" * 40)
print(f"{'Class':<15} {'Count':<10} {'Percentage'}")
print("-" * 40)
for idx in sorted(y_test.unique()):
    count = (y_test == idx).sum()
    pct = (count / len(y_test)) * 100
    label = {1: 'Diabetes', 2: 'No Diabetes', 3: 'Borderline'}[idx]
    print(f"{idx} ({label:<11}) {count:<10} {pct:.2f}%")
print("-" * 40)
print(f"{'Total':<15} {len(y_test):<10} 100.00%")

print()

# Define variable groups
lab_vars = ['Fasting_Glucose_mg/DL', 'Insulin_uU/mL', 'Glycohemoglobin_%', 'Albumin_urine_ug/mL']
binary_vars = ['HighBP_Ever', 'HighChol_Ever', 'On_CholMeds']
zero_fill_vars = ['Minutes_Vigorous_Activity', 'Freq_Vigorous_Activity']
median_vars = ['Freq_Moderate_Activity', 'Minutes_Moderate_Activity', 'Minutes_Sedentary']

# Create preprocessing pipeline with hybrid imputation (Geeks for Geeks)
preprocessor = ColumnTransformer([
    ('mice', IterativeImputer(random_state=42, max_iter=10), lab_vars),
    ('mode', SimpleImputer(strategy='most_frequent'), binary_vars),
    ('zero', SimpleImputer(strategy='constant', fill_value=0), zero_fill_vars),
    ('median', SimpleImputer(strategy='median'), median_vars)
])

# Full pipeline: impute -  scale - classify
# Uses class_weight='balanced' to handle class imbalance
# Dataset classes: 1=Diabetes (489), 2=No Diabetes (3079), 3=Borderline (132)
pipeline = Pipeline([
    ('imputer', preprocessor),
    ('scaler', StandardScaler()),
    ('svm', SVC(kernel='rbf', random_state=42, class_weight='balanced'))
])

# Fit on training data only
pipeline.fit(X_train, y_train)

# Evaluate
y_pred = pipeline.predict(X_test)

# Save model and test data for visualization
joblib.dump(pipeline, 'diabetes_svm_pipeline.pkl')
pd.DataFrame({'y_test': y_test, 'y_pred': y_pred}).to_csv('test_predictions.csv', index=False)
X_test.to_csv('X_test.csv', index=False)

print()
print("SVM RESULTS - HYBRID IMPUTATION (RESEARCH APPROACH)")
print()
print(f"\nAccuracy: {accuracy_score(y_test, y_pred):.4f}\n")
print(classification_report(y_test, y_pred, zero_division=0))

print()
print("Pipeline and test data saved for visualization!")
print()