import pandas as pd
import joblib

from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, r2_score


# ==========================================================
# 1. LOAD DATASET
# ==========================================================

print("Loading dataset...")

df = pd.read_csv("data/train.csv")

print("Dataset loaded successfully!")


# ==========================================================
# 2. SELECT FEATURES
# ==========================================================

features = [
    "OverallQual",
    "GrLivArea",
    "GarageCars",
    "TotalBsmtSF",
    "FullBath",
    "BedroomAbvGr",
    "YearBuilt"
]

target = "SalePrice"


# ==========================================================
# 3. CREATE MODELING DATA
# ==========================================================

data = df[features + [target]].copy()

print("\nSelected features:")
print(features)

print("\nTarget:")
print(target)


# ==========================================================
# 4. HANDLE MISSING VALUES
# ==========================================================

print("\nChecking missing values...")

print(data.isnull().sum())

data = data.dropna()

print("\nMissing rows removed.")
print("Remaining rows:", len(data))


# ==========================================================
# 5. SEPARATE INPUT AND OUTPUT
# ==========================================================

X = data[features]

y = data[target]


# ==========================================================
# 6. SPLIT DATA
# ==========================================================

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=42
)

print("\nTraining samples:", len(X_train))
print("Testing samples:", len(X_test))


# ==========================================================
# 7. CREATE MACHINE LEARNING MODEL
# ==========================================================

print("\nCreating Random Forest model...")

model = RandomForestRegressor(
    n_estimators=100,
    random_state=42,
    n_jobs=-1
)


# ==========================================================
# 8. TRAIN MODEL
# ==========================================================

print("Training model...")

model.fit(X_train, y_train)

print("Model training completed!")


# ==========================================================
# 9. MAKE PREDICTIONS
# ==========================================================

print("\nTesting model...")

predictions = model.predict(X_test)


# ==========================================================
# 10. EVALUATE MODEL
# ==========================================================

mae = mean_absolute_error(
    y_test,
    predictions
)

r2 = r2_score(
    y_test,
    predictions
)


print("\n==========================================")
print("       ESTATEAI MODEL PERFORMANCE")
print("==========================================")

print(f"Mean Absolute Error : ${mae:,.2f}")

print(f"R² Score            : {r2:.4f}")

print("==========================================")


# ==========================================================
# 11. SAVE TRAINED MODEL
# ==========================================================

model_path = "model/house_price_model.pkl"

joblib.dump(
    model,
    model_path
)

print("\n==========================================")
print("Model saved successfully!")
print(f"Location: {model_path}")
print("==========================================")