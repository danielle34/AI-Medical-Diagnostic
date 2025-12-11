import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import shap
from sklearn.experimental import enable_iterative_imputer
from sklearn.impute import IterativeImputer
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (accuracy_score, confusion_matrix, classification_report,
                             roc_curve, roc_auc_score, precision_recall_curve, average_precision_score)
from imblearn.over_sampling import SMOTE
from imblearn.pipeline import Pipeline as ImbPipeline

# LOAD & CLEAN
file_path = r'C:\Users\Joseph\Documents\Python\lab_q_data.xlsx'
try:
    df = pd.read_excel(file_path)
except FileNotFoundError:
    exit(f"Error: File not found at {file_path}")

# MICE Imputation
imputable_cols = [c for c in df.select_dtypes(include=np.number).columns if c != 'SEQN']
ranges = {c: (df[c].min(), df[c].max()) for c in imputable_cols}

mice = IterativeImputer(max_iter=10, random_state=42, min_value=0)
df_clean = df.copy()
df_clean[imputable_cols] = mice.fit_transform(df_clean[imputable_cols])

for c, (min_v, max_v) in ranges.items():
    df_clean[c] = df_clean[c].clip(lower=min_v, upper=max_v)

# Binary Constraints
for c in ['HighBP_Ever', 'HighChol_Ever', 'On_CholMeds', 'Diabetes_Risk_Flag']:
    if c in df_clean: df_clean[c] = df_clean[c].round(0).astype(int)
if 'Diabetes' in df_clean:
    # 1=Diabetes, 2=Healthy. We map 1->1 (Positive), 2->0 (Negative)
    df_clean['Diabetes'] = df_clean['Diabetes'].round(0).astype(int).clip(1, 2).replace({1: 1, 2: 0})

# SETUP
target = df_clean['Diabetes']
features = df_clean.drop(columns=['SEQN', 'Diabetes', 'Diabetes_Risk_Flag'])

X_train, X_test, y_train, y_test = train_test_split(
    features, target, test_size=0.2, random_state=42, stratify=target
)

pipeline = ImbPipeline([
    ('smote', SMOTE(random_state=42)),
    ('scale', StandardScaler()),
    ('model', LogisticRegression(solver='liblinear', random_state=42, max_iter=2000))
])

grid = GridSearchCV(pipeline, {'model__C': [0.01, 0.1, 1, 10, 100], 'model__penalty': ['l1', 'l2']},
                    cv=10, scoring='roc_auc', n_jobs=-1)
grid.fit(X_train, y_train)
best = grid.best_estimator_

# EVAL
y_pred = best.predict(X_test)
y_prob = best.predict_proba(X_test)[:, 1]

# Confusion Matrix
plt.figure(figsize=(8, 6))
# Switched labels back to [0, 1] so Healthy is first, Diabetes is second
cm = confusion_matrix(y_test, y_pred, labels=[0, 1])
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
            xticklabels=['Healthy', 'Diabetes'], yticklabels=['Healthy', 'Diabetes'])
plt.title('Confusion Matrix');
plt.ylabel('Actual Label')
plt.xlabel('Predicted Label')
plt.show()

# ROC Curve
fpr, tpr, _ = roc_curve(y_test, y_prob)
plt.figure(figsize=(8, 6))
plt.plot(fpr, tpr, color='darkorange', lw=2, label=f'AUC = {roc_auc_score(y_test, y_prob):.2f}')
plt.plot([0, 1], [0, 1], 'navy', linestyle='--')
plt.title('ROC Curve');
plt.legend();
plt.show()

# Precision-Recall
prec, rec, _ = precision_recall_curve(y_test, y_prob)
plt.figure(figsize=(8, 6))
plt.plot(rec, prec, color='purple', lw=2, label=f'AP = {average_precision_score(y_test, y_prob):.2f}')
plt.title('Precision-Recall');
plt.legend();
plt.show()

# FEATURE IMPORTANCE (SHAP Bar Plot Style)

try:
    print("Calculating SHAP values for Feature Importance...")
    # Prepare data for SHAP (Transforming via the scaler used in pipeline)
    pre = ImbPipeline([('scale', best.named_steps['scale'])])
    X_test_tr = pd.DataFrame(pre.transform(X_test), columns=features.columns)
    X_train_tr = pd.DataFrame(pre.transform(X_train), columns=features.columns)  # Background data

    # SHAP Values
    explainer = shap.LinearExplainer(best.named_steps['model'], X_train_tr)
    shap_vals = explainer(X_test_tr)

    # Calculate Mean Absolute SHAP Value for the Bar Chart
    shap_importance = np.abs(shap_vals.values).mean(axis=0)

    # Create DataFrame for Plotting
    imp = pd.DataFrame({
        'Feature': features.columns,
        'Importance': shap_importance
    })
    imp = imp.sort_values('Importance', ascending=False).head(15)

    # PLOT: Horizontal Bar Chart in SHAP Blue
    plt.figure(figsize=(10, 8))
    sns.barplot(x='Importance', y='Feature', data=imp, color='#008bfb')  # SHAP standard blue

    plt.title('Feature Importance (Mean |SHAP Value|)', fontsize=14)
    plt.xlabel('mean(|SHAP value|) (average impact on model output magnitude)', fontsize=12)
    plt.ylabel('')
    plt.grid(axis='x', linestyle='--', alpha=0.5)
    plt.subplots_adjust(left=0.35)  # Prevent labels from cutting off
    plt.show()

    # SHAP Summary Plot (Beeswarm)
    plt.figure(figsize=(16, 8))
    shap.summary_plot(shap_vals, X_test_tr, show=False)
    plt.subplots_adjust(left=0.45, right=0.9, top=0.9, bottom=0.1)
    plt.show()

    # SHAP Waterfall (First Patient)
    plt.figure(figsize=(16, 8))
    shap.plots.waterfall(shap_vals[0], show=False)
    plt.subplots_adjust(left=0.45, right=0.9)
    plt.show()

except Exception as e:
    print(f"SHAP Error: {e}")
    # Fallback to Coefficients if SHAP fails
    print("Falling back to standard coefficients...")
    imp = pd.DataFrame({'Feature': features.columns, 'Importance': best.named_steps['model'].coef_[0]})
    imp['Abs'] = imp['Importance'].abs()
    imp = imp.sort_values('Abs', ascending=False).head(15)
    plt.figure(figsize=(14, 8))
    sns.barplot(x='Importance', y='Feature', data=imp, palette='viridis')
    plt.title('Top 15 Feature Importances (Coefficients)')
    plt.show()

# Metrics
print("\n=== METRICS ===")
rep = classification_report(y_test, y_pred, output_dict=True)
print(f"Accuracy:  {accuracy_score(y_test, y_pred):.4f}")
print(f"Precision: {rep['weighted avg']['precision']:.4f}")
print(f"Recall:    {rep['weighted avg']['recall']:.4f}")
print(f"ROC AUC:   {roc_auc_score(y_test, y_prob):.4f}")
print(f"F1 Score:  {rep['weighted avg']['f1-score']:.4f}")
