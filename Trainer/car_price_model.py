import pandas as pd
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.preprocessing import LabelEncoder, StandardScaler
import joblib
import os

# Load the dataset
df = pd.read_csv("Data\processes2.csv")

print(df.head())
print(df.info())

# Rename columns for consistency
df.rename(columns={
    "selling_price": "price",
    "km_driven": "km",
    "seller_type": "type"
}, inplace=True)

# Extract brand from name
if "name" in df.columns:
    df["brand"] = df["name"].astype(str).apply(lambda x: x.split()[0] if len(x.split()) > 0 else "Unknown")

# Handle missing values
df.dropna(subset=["price", "km", "fuel", "type", "transmission", "owner", "seats", "brand"], inplace=True)

# Encode categorical columns
categorical_cols = ["fuel", "type", "transmission", "owner", "brand"]
label_encoders = {}

for col in categorical_cols:
    le = LabelEncoder()
    df[col] = le.fit_transform(df[col])
    label_encoders[col] = le

# Scale and save the scaler
scaler = StandardScaler()
df[["km", "year"]] = scaler.fit_transform(df[["km", "year"]])
joblib.dump(scaler, "models/scaler.pkl")

# Features and target
X = df[["year", "km", "fuel", "type", "transmission", "owner", "seats", "brand"]]
y = df["price"]

# Split into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# Create and train Random Forest model
model = RandomForestRegressor(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

# Predict and evaluate
y_pred = model.predict(X_test)
mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

print(f"MSE: {mse:.2f}")
print(f"R² Score: {r2:.2f}")

# Plot predictions vs actual
plt.scatter(y_test, y_pred)
plt.xlabel("Actual Prices")
plt.ylabel("Predicted Prices")
plt.title("Actual vs Predicted Car Prices")
plt.grid(True)
plt.tight_layout()
plt.show()

# Save the model and encoders
os.makedirs("models", exist_ok=True)
joblib.dump(model, "models/car_price_model.pkl")
joblib.dump(label_encoders, "models/encoders.pkl")
print("✅ Model and encoders saved.")

# Check the transformed training data
print(X_train.head())
