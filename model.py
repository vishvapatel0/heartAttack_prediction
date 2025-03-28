import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.metrics import accuracy_score, classification_report
from sklearn.utils import resample

# Load dataset
file_path = "/kaggle/input/cadasfsdzcfd/heart_attack_youngsters_india_final.csv"  # Update with your actual path
df = pd.read_csv(file_path)

# Encode categorical variables
categorical_cols = df.select_dtypes(include=['object']).columns.tolist()
categorical_cols.remove("Blood Pressure (systolic/diastolic mmHg)")
categorical_cols.remove("Heart Attack Likelihood")

label_encoders = {}
for col in categorical_cols:
    le = LabelEncoder()
    df[col] = le.fit_transform(df[col])
    label_encoders[col] = le

# Extract Systolic & Diastolic Blood Pressure
df[['Systolic_BP', 'Diastolic_BP']] = df["Blood Pressure (systolic/diastolic mmHg)"].str.split('/', expand=True).astype(float)
df.drop(columns=["Blood Pressure (systolic/diastolic mmHg)"], inplace=True)

# Encode target variable
df["Heart Attack Likelihood"] = df["Heart Attack Likelihood"].map({"Yes": 1, "No": 0})

# Handle class imbalance using upsampling
df_majority = df[df["Heart Attack Likelihood"] == 0]
df_minority = df[df["Heart Attack Likelihood"] == 1]

df_minority_upsampled = resample(df_minority, replace=True, n_samples=len(df_majority), random_state=42)
df_balanced = pd.concat([df_majority, df_minority_upsampled])

# Define features and target
X = df_balanced.drop(columns=["Heart Attack Likelihood"])
y = df_balanced["Heart Attack Likelihood"]

# Split into training and testing sets (80% train, 20% test)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Standardize numerical features
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# Hyperparameter tuning for Random Forest
param_grid = {
    'n_estimators': [100, 200, 300],
    'max_depth': [10, 20, 30, None],
    'min_samples_split': [2, 5, 10],
    'min_samples_leaf': [1, 2, 4]
}

rf = RandomForestClassifier(random_state=42)
grid_search = GridSearchCV(rf, param_grid, cv=5, scoring='accuracy', n_jobs=-1)
grid_search.fit(X_train, y_train)

# Best model
best_rf = grid_search.best_estimator_
best_rf.fit(X_train, y_train)

# Predictions
y_pred = best_rf.predict(X_test)

# Evaluate Model
accuracy = accuracy_score(y_test, y_pred)
report = classification_report(y_test, y_pred)

print(f"Optimized Model Accuracy: {accuracy:.2f}")
print("Classification Report:\n", report)



# -------------------------------------------
# Overfitting and Underfitting Analysis
# -------------------------------------------

# 1. Training vs. Testing Accuracy
train_accuracy = accuracy_score(y_train, best_rf.predict(X_train))
test_accuracy = accuracy_score(y_test, y_pred)

print(f"Training Accuracy: {train_accuracy:.2f}")
print(f"Testing Accuracy: {test_accuracy:.2f}")

if train_accuracy - test_accuracy > 0.1:
    print("Warning: The model may be overfitting.")
elif train_accuracy < 0.7 and test_accuracy < 0.7:
    print("Warning: The model may be underfitting.")
else:
    print("Model seems to have a good balance between bias and variance.")

# 2. Cross-Validation Performance
cv_scores = cross_val_score(best_rf, X_train, y_train, cv=5, scoring='accuracy')

print(f"Cross-Validation Mean Accuracy: {cv_scores.mean():.2f}")
print(f"Cross-Validation Standard Deviation: {cv_scores.std():.2f}")

if cv_scores.mean() < 0.7:
    print("Potential underfitting detected.")
elif train_accuracy - cv_scores.mean() > 0.1:
    print("Potential overfitting detected.")

# 3. Feature Importance
feature_importances = best_rf.feature_importances_
features = X.columns

plt.figure(figsize=(10, 5))
plt.barh(features, feature_importances)
plt.xlabel("Feature Importance")
plt.ylabel("Feature")
plt.title("Feature Importance in Random Forest")
plt.show()

# 4. Learning Curve Analysis
train_sizes, train_scores, test_scores = learning_curve(
    best_rf, X_train, y_train, cv=5, scoring="accuracy", train_sizes=np.linspace(0.1, 1.0, 5)
)

train_mean = train_scores.mean(axis=1)
test_mean = test_scores.mean(axis=1)

plt.plot(train_sizes, train_mean, label="Training Accuracy", marker='o')
plt.plot(train_sizes, test_mean, label="Validation Accuracy", marker='o')
plt.xlabel("Training Set Size")
plt.ylabel("Accuracy")
plt.legend()
plt.title("Learning Curve")
plt.show()

# -------------------------------------------
# Save Model, Scaler, and Label Encoder
# -------------------------------------------

# Save the trained model
joblib.dump(best_rf, "random_forest_heart_attack.pkl")
print("Model saved as random_forest_heart_attack.pkl")

# Save the scaler for preprocessing in future predictions
joblib.dump(scaler, "scaler.pkl")
print("Scaler saved as scaler.pkl")

# Save the label encoders
joblib.dump(label_encoders, "label_encoders.pkl")
print("Label Encoders saved as label_encoders.pkl")
