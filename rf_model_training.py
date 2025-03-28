import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, KFold, cross_val_score
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report
from joblib import dump

df = pd.read_csv('dataset_dg.csv')
df = df.drop('temp_C', axis=1)
df = df.rename(columns={col: col.replace('_seq', '_dg') for col in df.columns if col.endswith('_seq')})

class_col = 'class_2'

X = df.drop(columns=[class_col])
y = df[class_col]

X.replace([np.inf, -np.inf], np.nan, inplace=True)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=100)

rf_model = RandomForestClassifier(
    n_estimators=250,
    max_depth=None,
    min_samples_split=2,
    min_samples_leaf=1,
    random_state=42
)

k = 5  
kf = KFold(n_splits=k, shuffle=True, random_state=100)
cv_scores = cross_val_score(rf_model, X_train, y_train, cv=kf, scoring="accuracy")

print(f"K-Fold Cross-Validation Scores: {cv_scores}")
print(f"Mean CV Accuracy: {np.mean(cv_scores):.4f}")
print(f"Standard Deviation: {np.std(cv_scores):.4f}\n")

rf_model.fit(X_train, y_train)

y_pred = rf_model.predict(X_test)

print("Final Test Set Accuracy:", accuracy_score(y_test, y_pred))
print("\nClassification Report:")
print(classification_report(y_test, y_pred))

feature_importance = pd.DataFrame({
    'feature': X.columns,
    'importance': rf_model.feature_importances_
}).sort_values('importance', ascending=False)

dump(rf_model, 'rf_model.joblib')