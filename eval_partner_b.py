"""
Partner B: Stratified/Time-Aware + 5-Fold Specialized CV
Author: Brett Coleman
Dataset: Wine Quality (Multiclass)
Metric: Accuracy
"""

from sklearn.datasets import load_wine
data = load_wine(as_frame=True)
X, y = data.data, data.target

from sklearn.model_selection import train_test_split, StratifiedKFold, cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
import numpy as np
import os
import pandas as pd

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, stratify=y, random_state=42)

# Create a pipeline with scaling and logistic regression for better performance on data set
pipe = Pipeline([
    ('scaler', StandardScaler()),
    ('logreg', LogisticRegression(max_iter=1000))
])

# Add cross-validation for StratifiedKFold
cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
cv_scores = cross_val_score(pipe, X_train, y_train, cv=cv, scoring='accuracy')

# Fit the data
pipe.fit(X_train, y_train)

# Predict on the test set
y_pred = pipe.predict(X_test)
print(f"Predictions on test set: {y_pred}")

# Metric evaluation on cross validation and test set
print(f"Cross-validation accuracy scores: {cv_scores.round(4)}")
print(f"Mean cross-validation accuracy: {round(np.mean(cv_scores)),4}")
print(f"Standard deviation cross-validation accuracy: {round(np.std(cv_scores),4)}")

# Test set accuracy
accuracy_score(y_test, pipe.predict(X_test))
print(f"Accuracy: {round(accuracy_score(y_test, pipe.predict(X_test)),4)}")

# Save scoores to comparison file - Append to existing file or create new one!
file = "comparison.csv"

# Check if file exists or is empty to write header
write_header = not os.path.exists(file) or os.path.getsize(file) == 0

# Build one row of data
df = pd.DataFrame([{
    "strategy": "stratified",
    "mean_cv_accuracy": round(np.mean(cv_scores),4),
    "std_cv_accuracy": round(np.std(cv_scores),4),
    "test_accuracy": round(accuracy_score(y_test, pipe.predict(X_test)),4)
}])

# Append to CSV file
df.to_csv(file, mode="a", header=write_header, index=False)

    