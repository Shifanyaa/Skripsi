from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score
from sklearn.model_selection import cross_val_score
from sklearn.preprocessing import StandardScaler
import pandas as pd

# Load your data
train_data_balance = pd.read_csv("train_data_balanced.csv")
val_data = pd.read_csv("val_data.csv")
test_data = pd.read_csv("test_data.csv")

# Select numeric features
feature_cols = ['LogP', 'TPSA', 'hbond_acceptors', 'hbond_donors', 
                'num_atoms', 'num_bonds', 'rotatable_bonds', 'weight']

X_train = train_data_balance[feature_cols]
y_train = train_data_balance['Label']
X_val = val_data[feature_cols]
y_val = val_data['Label']
X_test = test_data[feature_cols]
y_test = test_data['Label']

# Scale the data
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_val_scaled = scaler.transform(X_val)
X_test_scaled = scaler.transform(X_test)

# RANDOM FOREST
rf = RandomForestClassifier(n_estimators=100, class_weight="balanced", random_state=0)
rf.fit(X_train_scaled, y_train)

print("Random Forest:")
print("Validation Confusion Matrix:\n", confusion_matrix(y_val, rf.predict(X_val_scaled)))
print("Test Confusion Matrix:\n", confusion_matrix(y_test, rf.predict(X_test_scaled)))
print("Validation AUC:", roc_auc_score(y_val, rf.predict_proba(X_val_scaled)[:, 1]))
print("Test AUC:", roc_auc_score(y_test, rf.predict_proba(X_test_scaled)[:, 1]))
print("Validation Report:\n", classification_report(y_val, rf.predict(X_val_scaled)))
print("Test Report:\n", classification_report(y_test, rf.predict(X_test_scaled)))

# LOGISTIC REGRESSION
lr = LogisticRegression(class_weight="balanced", max_iter=1000, random_state=0)
lr.fit(X_train_scaled, y_train)

print("\nLogistic Regression:")
print("Validation Confusion Matrix:\n", confusion_matrix(y_val, lr.predict(X_val_scaled)))
print("Test Confusion Matrix:\n", confusion_matrix(y_test, lr.predict(X_test_scaled)))
print("Validation AUC:", roc_auc_score(y_val, lr.predict_proba(X_val_scaled)[:, 1]))
print("Test AUC:", roc_auc_score(y_test, lr.predict_proba(X_test_scaled)[:, 1]))
print("Validation Report:\n", classification_report(y_val, lr.predict(X_val_scaled)))
print("Test Report:\n", classification_report(y_test, lr.predict(X_test_scaled)))

# Cross-validation for logistic regression
cv_scores = cross_val_score(lr, X_train_scaled, y_train, cv=5)
print("\nCross-validation scores:", cv_scores)
print("Mean CV accuracy:", cv_scores.mean())
