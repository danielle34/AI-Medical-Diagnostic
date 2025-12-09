"""
SVM Visualization Script
"""
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from sklearn.metrics import confusion_matrix, precision_recall_fscore_support
from sklearn.inspection import permutation_importance
import joblib

# Load saved data
predictions = pd.read_csv('test_predictions.csv')
y_test = predictions['y_test']
y_pred = predictions['y_pred']
X_test = pd.read_csv('X_test.csv')
pipeline = joblib.load('diabetes_svm_pipeline.pkl')

# Class names
classes = ['Diabetes', 'No Diabetes', 'Borderline']

# 1. Confusion Matrix
print("Confusion Matrix")
cm = confusion_matrix(y_test, y_pred)
plt.figure(figsize=(8, 6))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
            xticklabels=classes, yticklabels=classes)
plt.title('Confusion Matrix - SVM Classification', fontsize=14, fontweight='bold')
plt.ylabel('Actual', fontsize=12)
plt.xlabel('Predicted', fontsize=12)
plt.tight_layout()
plt.savefig('confusion_matrix.png', dpi=300, bbox_inches='tight')
plt.show()

# 2. Classification Metrics
print("Classification Metrics")
precision, recall, f1, _ = precision_recall_fscore_support(y_test, y_pred, zero_division=0)
x = np.arange(len(classes))
width = 0.25

fig, ax = plt.subplots(figsize=(12, 6))
ax.bar(x - width, precision, width, label='Precision', color='skyblue')
ax.bar(x, recall, width, label='Recall', color='lightcoral')
ax.bar(x + width, f1, width, label='F1-Score', color='lightgreen')
ax.set_xlabel('Class', fontsize=12)
ax.set_ylabel('Score', fontsize=12)
ax.set_title('Classification Metrics by Class', fontsize=14, fontweight='bold')
ax.set_xticks(x)
ax.set_xticklabels(classes)
ax.legend(fontsize=11)
ax.set_ylim([0, 1.1])
ax.grid(axis='y', alpha=0.3, linestyle='--')
plt.tight_layout()
plt.savefig('classification_metrics.png', dpi=300, bbox_inches='tight')
plt.show()

# 3. Actual vs Predicted Distribution
print(" Actual vs Predicted Distribution")
actual_counts = [sum(y_test == 1), sum(y_test == 2), sum(y_test == 3)]
predicted_counts = [sum(y_pred == 1), sum(y_pred == 2), sum(y_pred == 3)]
x = np.arange(len(classes))
width = 0.35

fig, ax = plt.subplots(figsize=(10, 6))
ax.bar(x - width/2, actual_counts, width, label='Actual', color='steelblue')
ax.bar(x + width/2, predicted_counts, width, label='Predicted', color='coral')
ax.set_xlabel('Class', fontsize=12)
ax.set_ylabel('Count', fontsize=12)
ax.set_title('Actual vs Predicted Class Distribution', fontsize=14, fontweight='bold')
ax.set_xticks(x)
ax.set_xticklabels(classes)
ax.legend(fontsize=11)
ax.grid(axis='y', alpha=0.3, linestyle='--')
plt.tight_layout()
plt.savefig('actual_vs_predicted.png', dpi=300, bbox_inches='tight')
plt.show()

# 4. Feature Importance
perm_importance = permutation_importance(pipeline, X_test, y_test,
                                         n_repeats=10, random_state=42)
importance_df = pd.DataFrame({
    'Feature': X_test.columns,
    'Importance': perm_importance.importances_mean
}).sort_values('Importance', ascending=False)

plt.figure(figsize=(10, 8))
plt.barh(importance_df['Feature'], importance_df['Importance'], color='teal')
plt.xlabel('Permutation Importance', fontsize=12)
plt.title('Feature Importance - SVM Model', fontsize=14, fontweight='bold')
plt.gca().invert_yaxis()
plt.grid(axis='x', alpha=0.3, linestyle='--')
plt.tight_layout()
plt.savefig('feature_importance.png', dpi=300, bbox_inches='tight')
plt.show()

print("\n" + "=" * 60)
print("List of Visualizations")
print("  - confusion_matrix.png")
print("  - classification_metrics.png")
print("  - actual_vs_predicted.png")
print("  - feature_importance.png")
print("=" * 60)