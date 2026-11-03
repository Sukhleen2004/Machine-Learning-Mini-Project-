
import os
import joblib
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor

from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

# Load dataset
data = pd.read_csv("Student_Performance.csv")

# Input features
features = [
    "StudyHours",
    "Attendance",
    "PreviousMarks",
    "AssignmentScore",
    "SleepHours"
]

X = data[features]
y = data["FinalMarks"]

# Split data
X_train, X_test, y_train, y_test = train_test_split(
    X, y,
    test_size=0.2,
    random_state=42
)

# Define algorithms
algorithms = {
    "Linear Regression": LinearRegression(),

    "Decision Tree": DecisionTreeRegressor(
        max_depth=5,
        random_state=42
    ),

    "Random Forest": RandomForestRegressor(
        n_estimators=100,
        max_depth=6,
        random_state=42
    )
}

models = {}
results = []

# Train and evaluate
for name, algorithm in algorithms.items():

    pipeline = Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler()),
        ("model", algorithm)
    ])

    pipeline.fit(X_train, y_train)

    predictions = pipeline.predict(X_test)

    mse = mean_squared_error(y_test, predictions)
    mae = mean_absolute_error(y_test, predictions)
    r2 = r2_score(y_test, predictions)

    models[name] = pipeline

    results.append({
        "Model": name,
        "MSE": round(mse, 2),
        "MAE": round(mae, 2),
        "R2": round(r2, 3)
    })

    print("\nModel:", name)
    print("MSE:", round(mse, 2))
    print("MAE:", round(mae, 2))
    print("R2 Score:", round(r2, 3))

# Select model with lowest MAE
best_model_name = min(
    results,
    key=lambda item: item["MAE"]
)["Model"]

print("\nBest Model:", best_model_name)

# Save models and results
os.makedirs("model", exist_ok=True)

joblib.dump({
    "models": models,
    "best_model": best_model_name,
    "results": results,
    "features": features,
    "total_students": len(data),
    "average_marks": round(float(y.mean()), 2)
}, "model/student_model.pkl")

print("\nModels saved successfully!")
