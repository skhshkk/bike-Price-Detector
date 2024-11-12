# training.py
import pandas as pd
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.model_selection import cross_val_score
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from xgboost import XGBRegressor
from sklearn.svm import SVR
import pickle

# Load the data
df = pd.read_csv("BIKE DETAILS.csv")

# Drop rows with any missing values
df = df.dropna()

# Convert columns to appropriate types and filter invalid values
df["year"] = pd.to_numeric(df["year"], errors="coerce")
df["selling_price"] = pd.to_numeric(df["selling_price"], errors="coerce")
df["km_driven"] = pd.to_numeric(df["km_driven"], errors="coerce")
df = df[(df["selling_price"] > 0) & (df["km_driven"] > 0)]

# Split data into features (X) and target (y)
X = df.drop("selling_price", axis=1)
y = df["selling_price"]

# Define numerical and categorical transformers
num_features = X.select_dtypes(exclude="object").columns
cat_features = X.select_dtypes(include="object").columns

num_transformer = StandardScaler()
cat_transformer = OneHotEncoder(handle_unknown="ignore")

# Preprocessor pipeline
preprocessor = ColumnTransformer(
    transformers=[
        ("num", num_transformer, num_features),
        ("cat", cat_transformer, cat_features),
    ]
)

# Initialize models
models = {
    "Linear Regression": LinearRegression(),
    "Random Forest": RandomForestRegressor(),
    "XGBoost": XGBRegressor(),
    "Support Vector Regressor": SVR(),
}

# Train models and evaluate performance
best_score = float("-inf")
best_model = None

for name, model in models.items():
    # Create pipeline for each model
    pipe = Pipeline(steps=[("preprocessor", preprocessor), ("model", model)])
    scores = cross_val_score(
        pipe, X, y, cv=5, scoring="r2"
    )  # Use R-squared as the score
    score_mean = scores.mean()
    print(f"{name} R-squared mean score: {score_mean:.4f}")

    # Select the best model
    if score_mean > best_score:
        best_score = score_mean
        best_model = pipe

# Fit the best model to entire dataset
best_model.fit(X, y)

# Save the best model and categorical encodings
with open("best_model.pkl", "wb") as f:
    pickle.dump(best_model, f)

print("Training complete. Best model saved as 'best_model.pkl'.")
