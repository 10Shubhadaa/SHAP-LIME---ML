# LIME-like local surrogate implementation (no external LIME package required)
import numpy as np
import pandas as pd
from sklearn.datasets import load_breast_cancer
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.linear_model import Ridge
from sklearn.metrics import accuracy_score
import matplotlib.pyplot as plt
from sklearn.metrics import pairwise_distances

# Load data
data = load_breast_cancer()
X = pd.DataFrame(data.data, columns=data.feature_names)
y = data.target

# Train/test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

# Train RandomForest
clf = RandomForestClassifier(n_estimators=200, random_state=42)
clf.fit(X_train, y_train)
y_pred = clf.predict(X_test)
acc = accuracy_score(y_test, y_pred)

# Choose an instance to explain (index in X_test)
instance_idx = 3
x0 = X_test.iloc[instance_idx:instance_idx+1].values.flatten()

# Create perturbations
np.random.seed(0)
num_samples = 1000
X_perturb = np.tile(x0, (num_samples,1)).astype(float)
feature_std = X_train.std().values
for i in range(X_perturb.shape[1]):
    X_perturb[:,i] += np.random.normal(0, 0.3*feature_std[i], size=num_samples)

# Get model predictions (probability for class 1)
probs = clf.predict_proba(X_perturb)[:,1]

# Compute distances and weights (exponential kernel)
dists = pairwise_distances(X_perturb, x0.reshape(1,-1)).flatten()
kernel_width = np.sqrt(X.shape[1]) * 0.75
weights = np.sqrt(np.exp(-(dists**2) / (kernel_width**2)))

# Fit weighted linear model (Ridge) as surrogate
model_surrogate = Ridge(alpha=1.0)
model_surrogate.fit(X_perturb, probs, sample_weight=weights)
coefs = model_surrogate.coef_

# Prepare explanation
feature_names = list(X.columns)
coef_df = pd.DataFrame({'feature': feature_names, 'coef': coefs})
coef_df['abs'] = coef_df['coef'].abs()
coef_df = coef_df.sort_values('abs', ascending=False).head(12).sort_values('coef', ascending=True)

# Print results and plot
print("RandomForest test accuracy:", round(acc,4))
print("Explaining test instance index:", instance_idx)
print(coef_df[['feature','coef']])

plt.figure(figsize=(8,6))
plt.barh(coef_df['feature'], coef_df['coef'])
plt.xlabel('Surrogate coefficient (positive -> increases probability of class 1)')
plt.title('LIME-like Local Surrogate Explanation (Breast Cancer instance)')
plt.tight_layout()
plt.show()
