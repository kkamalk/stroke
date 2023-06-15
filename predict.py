import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from imblearn.over_sampling import SMOTE
import joblib

# Load the preprocessed dataset
df = pd.read_csv("preprocessed_data.csv")

print (df.head())

# Split the data into training and test sets
X = df.drop("stroke", axis=1)
y = df["stroke"]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Apply SMOTE for class imbalance
smote = SMOTE()
X_resampled, y_resampled = smote.fit_resample(X_train, y_train)

# Standardize the features
scaler = StandardScaler()
X_resampled = scaler.fit_transform(X_resampled)
X_test = scaler.transform(X_test)

# Train the Random Forest classifier
r_best_model = RandomForestClassifier(max_depth=20, min_samples_split=5, n_estimators=200, random_state=42)
r_best_model.fit(X_resampled, y_resampled)

# Save the trained model
joblib.dump(r_best_model,"best_model.pkl")
