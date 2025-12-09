"""
SVM Decision Boundary Visualization
Shows non-linear decision boundaries created by RBF kernel
Used Chat GPT and Google Gemini check and clean code)
"""
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
import joblib


pipeline = joblib.load('diabetes_svm_pipeline.pkl')
predictions = pd.read_csv('test_predictions.csv')
X_test = pd.read_csv('X_test.csv')
y_test = predictions['y_test']
y_pred = predictions['y_pred']


print("Creating 2D PCA visualization...")

# Apply the preprocessing from the pipeline (imputation + scaling)
X_test_preprocessed = pipeline.named_steps['imputer'].transform(X_test)
X_test_scaled = pipeline.named_steps['scaler'].transform(X_test_preprocessed)

# Reduce to 2D using PCA
pca = PCA(n_components=2)
X_test_pca = pca.fit_transform(X_test_scaled)

print(f"PCA explains {pca.explained_variance_ratio_.sum() * 100:.1f}% of variance")

# Create a mesh grid for decision boundary
h = 0.02  # Step size in the mesh
x_min, x_max = X_test_pca[:, 0].min() - 1, X_test_pca[:, 0].max() + 1
y_min, y_max = X_test_pca[:, 1].min() - 1, X_test_pca[:, 1].max() + 1
xx, yy = np.meshgrid(np.arange(x_min, x_max, h),
                     np.arange(y_min, y_max, h))

# Train a new SVM on PCA-reduced data to visualize boundaries
from sklearn.svm import SVC

svm_2d = SVC(kernel='rbf', random_state=42, class_weight='balanced')
svm_2d.fit(X_test_pca, y_test)

# Predict on mesh grid
Z = svm_2d.predict(np.c_[xx.ravel(), yy.ravel()])
Z = Z.reshape(xx.shape)

# Plot decision boundaries
plt.figure(figsize=(14, 6))

# Subplot 1: Decision Boundaries
plt.subplot(1, 2, 1)
plt.contourf(xx, yy, Z, alpha=0.3, cmap='RdYlBu', levels=[0.5, 1.5, 2.5, 3.5])
plt.contour(xx, yy, Z, colors='black', linewidths=0.5, levels=[1.5, 2.5])

# Plot actual data points
colors = {1: 'red', 2: 'blue', 3: 'orange'}
labels = {1: 'Diabetes', 2: 'No Diabetes', 3: 'Borderline'}
for class_val in [1, 2, 3]:
    mask = y_test == class_val
    plt.scatter(X_test_pca[mask, 0], X_test_pca[mask, 1],
                c=colors[class_val], label=labels[class_val],
                edgecolors='black', s=50, alpha=0.7)

plt.xlabel(f'First Principal Component ({pca.explained_variance_ratio_[0] * 100:.1f}% variance)', fontsize=12)
plt.ylabel(f'Second Principal Component ({pca.explained_variance_ratio_[1] * 100:.1f}% variance)', fontsize=12)
plt.title('SVM Decision Boundaries (Non-Linear RBF Kernel)\nProjected to 2D via PCA',
          fontsize=14, fontweight='bold')
plt.legend(loc='best')
plt.grid(alpha=0.3)

# Subplot 2: Predictions vs Actual
plt.subplot(1, 2, 2)
# Plot predictions with different marker
for class_val in [1, 2, 3]:
    mask_correct = (y_test == class_val) & (y_pred == class_val)
    mask_wrong = (y_test == class_val) & (y_pred != class_val)

    plt.scatter(X_test_pca[mask_correct, 0], X_test_pca[mask_correct, 1],
                c=colors[class_val], marker='o', s=50, alpha=0.7,
                edgecolors='black', linewidths=1.5,
                label=f'{labels[class_val]} (Correct)')

    plt.scatter(X_test_pca[mask_wrong, 0], X_test_pca[mask_wrong, 1],
                c=colors[class_val], marker='x', s=100, alpha=0.9,
                linewidths=2,
                label=f'{labels[class_val]} (Wrong)')

plt.xlabel(f'First Principal Component', fontsize=12)
plt.ylabel(f'Second Principal Component', fontsize=12)
plt.title('SVM Predictions: Correct (○) vs Incorrect (✗)',
          fontsize=14, fontweight='bold')
plt.legend(loc='best', fontsize=9)
plt.grid(alpha=0.3)

plt.tight_layout()
plt.savefig('svm_decision_boundaries.png', dpi=300, bbox_inches='tight')
plt.show()


print("\nCreating visualization with top 2 features...")

# Get top 2 features from permutation importance
from sklearn.inspection import permutation_importance

perm_importance = permutation_importance(pipeline, X_test, y_test,
                                         n_repeats=10, random_state=42)
importance_df = pd.DataFrame({
    'Feature': X_test.columns,
    'Importance': perm_importance.importances_mean
}).sort_values('Importance', ascending=False)

top_2_features = importance_df.head(2)['Feature'].values
print(f"Top 2 features: {top_2_features[0]}, {top_2_features[1]}")

# Get indices of top 2 features
feature_indices = [X_test.columns.get_loc(f) for f in top_2_features]

# Extract these features after preprocessing
X_top2 = X_test_scaled[:, feature_indices]

# Create mesh grid
x_min, x_max = X_top2[:, 0].min() - 1, X_top2[:, 0].max() + 1
y_min, y_max = X_top2[:, 1].min() - 1, X_top2[:, 1].max() + 1
xx, yy = np.meshgrid(np.arange(x_min, x_max, 0.02),
                     np.arange(y_min, y_max, 0.02))

# Train SVM on top 2 features
svm_top2 = SVC(kernel='rbf', random_state=42, class_weight='balanced')
svm_top2.fit(X_top2, y_test)

# Predict on mesh
Z = svm_top2.predict(np.c_[xx.ravel(), yy.ravel()])
Z = Z.reshape(xx.shape)

# Plot
plt.figure(figsize=(12, 8))
plt.contourf(xx, yy, Z, alpha=0.3, cmap='RdYlBu', levels=[0.5, 1.5, 2.5, 3.5])
plt.contour(xx, yy, Z, colors='black', linewidths=1.5, levels=[1.5, 2.5])

# Plot data points
for class_val in [1, 2, 3]:
    mask = y_test == class_val
    plt.scatter(X_top2[mask, 0], X_top2[mask, 1],
                c=colors[class_val], label=labels[class_val],
                edgecolors='black', s=80, alpha=0.8, linewidths=1.5)

plt.xlabel(f'{top_2_features[0]} (Scaled)', fontsize=14, fontweight='bold')
plt.ylabel(f'{top_2_features[1]} (Scaled)', fontsize=14, fontweight='bold')
plt.title(f'Non-Linear SVM Decision Boundaries\nTop 2 Most Important Features',
          fontsize=16, fontweight='bold')
plt.legend(loc='best', fontsize=12)
plt.grid(alpha=0.3)
plt.tight_layout()
plt.savefig('svm_top2_features.png', dpi=300, bbox_inches='tight')
plt.show()


print("\nCreating Linear vs Non-Linear comparison...")

fig, axes = plt.subplots(1, 2, figsize=(16, 6))

# Use PCA-reduced data for comparison
for idx, (kernel, ax) in enumerate(zip(['linear', 'rbf'], axes)):
    # Train SVM with different kernels
    svm = SVC(kernel=kernel, random_state=42, class_weight='balanced')
    svm.fit(X_test_pca, y_test)

    # Create mesh
    h = 0.02
    x_min, x_max = X_test_pca[:, 0].min() - 1, X_test_pca[:, 0].max() + 1
    y_min, y_max = X_test_pca[:, 1].min() - 1, X_test_pca[:, 1].max() + 1
    xx, yy = np.meshgrid(np.arange(x_min, x_max, h),
                         np.arange(y_min, y_max, h))

    # Predict
    Z = svm.predict(np.c_[xx.ravel(), yy.ravel()])
    Z = Z.reshape(xx.shape)

    # Plot decision boundaries
    ax.contourf(xx, yy, Z, alpha=0.3, cmap='RdYlBu', levels=[0.5, 1.5, 2.5, 3.5])
    ax.contour(xx, yy, Z, colors='black', linewidths=1.5, levels=[1.5, 2.5])

    # Plot data
    for class_val in [1, 2, 3]:
        mask = y_test == class_val
        ax.scatter(X_test_pca[mask, 0], X_test_pca[mask, 1],
                   c=colors[class_val], label=labels[class_val],
                   edgecolors='black', s=50, alpha=0.7)

    # Calculate accuracy
    acc = svm.score(X_test_pca, y_test)

    kernel_name = 'LINEAR' if kernel == 'linear' else 'NON-LINEAR (RBF)'
    ax.set_xlabel('First Principal Component', fontsize=12)
    ax.set_ylabel('Second Principal Component', fontsize=12)
    ax.set_title(f'{kernel_name} Kernel\nAccuracy: {acc:.2%}',
                 fontsize=14, fontweight='bold')
    ax.legend(loc='best')
    ax.grid(alpha=0.3)

plt.tight_layout()
plt.savefig('svm_linear_vs_nonlinear.png', dpi=300, bbox_inches='tight')
plt.show()

print("\n" + "=" * 60)
print("SUMMARY")
print("=" * 60)
print(f"\nYour SVM uses RBF (Radial Basis Function) kernel")
print(f"This creates NON-LINEAR decision boundaries")
print(f"\nKey characteristics:")
print(f"  - Curved, complex boundaries (not straight lines)")
print(f"  - Can capture non-linear relationships")
print(f"  - More flexible than linear SVM")
print(f"  - Better for complex medical data")
print("\nVisualization files saved:")
print("  1. svm_decision_boundaries.png - PCA projection with boundaries")
print("  2. svm_top2_features.png - Top 2 features with boundaries")
print("  3. svm_linear_vs_nonlinear.png - Comparison of kernels")
print("=" * 60)