import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report, accuracy_score
from imblearn.over_sampling import SMOTE
import joblib

# Load the dataset
df = pd.read_csv('data/german_credit_data.csv')
print("Columns in dataset:", df.columns)

# Add dummy Risk column if not present
if 'Risk' not in df.columns:
    print("Adding dummy 'Risk' column...")
    df['Risk'] = np.random.randint(0, 2, df.shape[0])

# Drop unnecessary column
X = df.drop(['Unnamed: 0', 'Risk'], axis=1)
y = df['Risk']

# Convert categorical features using one-hot encoding
X = pd.get_dummies(X, drop_first=True)

# Feature scaling
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Handling class imbalance using SMOTE
smote = SMOTE(random_state=42)
X_resampled, y_resampled = smote.fit_resample(X_scaled, y)

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(X_resampled, y_resampled, test_size=0.2, random_state=42)
print(f"Training set size: {X_train.shape[0]}")
print(f"Testing set size: {X_test.shape[0]}")

# Hyperparameter tuning
param_grid = {
    'n_estimators': [100, 200],
    'max_depth': [10, 20],
    'min_samples_split': [2, 5],
}

grid = GridSearchCV(RandomForestClassifier(random_state=42), param_grid, cv=5, n_jobs=-1, verbose=1)
grid.fit(X_train, y_train)

# Best model
best_model = grid.best_estimator_

# Evaluate
y_pred = best_model.predict(X_test)
print("Accuracy Score:", accuracy_score(y_test, y_pred))
print("\nClassification Report:\n", classification_report(y_test, y_pred))

# Save the model
joblib.dump(best_model, 'app/model.pkl')
print("Model saved as 'model.pkl'")

# Save feature names for Streamlit
feature_names = X.columns.tolist()
joblib.dump(feature_names, 'app/feature_names.pkl')
print("Feature names saved as 'feature_names.pkl'")
