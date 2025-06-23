import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import (
    confusion_matrix, classification_report,
)
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from imblearn.pipeline import Pipeline  
import matplotlib.pyplot as plt
import seaborn as sns
from imblearn.over_sampling import SMOTE
from xgboost import XGBClassifier

# Load the dataset
df = pd.read_csv(r"c:\Users\aryan\Downloads\creditcard.csv")
df["Class"] = df["Class"].astype(int)

# Split features and target
X = df.drop(["Class", "Time"], axis=1)
y = df["Class"]

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# Pipeline setup
pipeline = Pipeline([
    ('scaler', StandardScaler()),
    ('smote',SMOTE(random_state=42)),
    ('ee',XGBClassifier(eval_metric='logloss',random_state=42))
])

# Grid search parameters
param_grid = {
    "ee__n_estimators": [50, 100],
    "ee__max_depth": [3, 6],
    "ee__learning_rate":[0.05,0.1]
}

# GridSearchCV setup
grid = GridSearchCV(pipeline, param_grid, cv=3, n_jobs=-1)

# Fit model
grid.fit(X_train, y_train)

# Predict
pred = grid.predict(X_test)

# Results
print("Best Parameters:", grid.best_params_)
print("Confusion Matrix:\n", confusion_matrix(y_test, pred))
print("Classification Report:\n", classification_report(y_test, pred))
from sklearn.metrics import roc_curve, auc
fpr, tpr, _ = roc_curve(y_test, grid.predict_proba(X_test)[:, 1])
plt.plot(fpr, tpr)
plt.title("ROC Curve")
plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.show()
