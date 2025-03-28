import pandas as pd
import pickle
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

# Load dataset
df = pd.read_csv("data/train.csv")

# Define features and target
FEATURE_COLUMNS = [
    "profile pic", "nums/length username", "fullname words",
    "nums/length fullname", "name==username", "description length",
    "external URL", "private", "#posts", "#followers", "#follows"
]
TARGET_COLUMN = "fake"

X = df[FEATURE_COLUMNS]
y = df[TARGET_COLUMN]

# Split data into train and test
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train model
model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

# Evaluate model
y_pred = model.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
print(f"Model Accuracy: {accuracy:.2f}")

# Save trained model
with open("models/fake_profile_detector.pkl", "wb") as model_file:
    pickle.dump(model, model_file)

print("Model saved successfully!")

