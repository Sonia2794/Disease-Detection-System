import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import pickle
import joblib

# Step 1: Load dataset
df = pd.read_csv("testing.csv", encoding='ISO-8859-1')

# Step 2: Features and labels
X = df.iloc[:, :-1]
y = df.iloc[:, -1]

# Step 3: Encode target labels
label_encoder = LabelEncoder()
y_encoded = label_encoder.fit_transform(y)

# Step 4: Split into train/test
X_train, X_test, y_train, y_test = train_test_split(
    X, y_encoded, test_size=0.2, random_state=42
)

# Step 5: Train model
model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

# Step 6: Evaluate accuracy
y_pred = model.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
print(f"✅ Model accuracy on test set: {accuracy * 100:.2f}%")

# Step 7: Save model and label encoder
with open("disease_model.pkl", "wb") as f:
    pickle.dump(model, f)

joblib.dump(label_encoder, "label_encoder.pkl")

print("✅ Model and label encoder saved.")
