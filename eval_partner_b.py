from sklearn.datasets import load_wine
data = load_wine(as_frame=True)
X, y = data.data, data.target

from sklearn.model_selection import train_test_split, StratifiedKFold, cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
import numpy as np


X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

# Create a pipeline with scaling and logistic regression for better performance on data set
pipe = Pipeline([
    ('scaler', StandardScaler()),
    ('logreg', LogisticRegression(max_iter=1000))
])

# Add cross-validation for StratifiedKFold
cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
cv_scores = cross_val_score(pipe, X_train, y_train,
                            cv=cv, scoring='accuracy')
# Fit the data
pipe.fit(X_train, y_train)

# Predict on the test set
pipe.predict(X_test)

# Metric evaluation on cross validation and test set
print(f"Cross-validation accuracy scores: {cv_scores}")
print(f"Mean cross-validation accuracy: {np.mean(cv_scores)}")
print(f"Standard deviation cross-validation accuracy: {np.std(cv_scores)}")

# Test set accuracy
accuracy_score(y_test, pipe.predict(X_test))
print(f"Accuracy: {accuracy_score(y_test, pipe.predict(X_test))}")