from sklearn.datasets import load_wine
data = load_wine(as_frame=True)
X, y = data.data, data.target

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression

pipe = Pipeline([
    ('scaler', StandardScaler()),
    ('logreg', LogisticRegression(max_iter=1000))
])

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)
pipe.fit(X_train, y_train)
pipe.predict(X_test)