
import os
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.datasets import fetch_california_housing
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

# Create folders for saving results
os.makedirs("data", exist_ok=True)
os.makedirs("outputs", exist_ok=True)

# Load dataset
print("\nLoading California Housing Dataset...")
housing = fetch_california_housing(as_frame=True)
df = housing.frame

print("\nDataset loaded successfully!")
print("Dataset shape:", df.shape)
print(df.head())

# Save dataset
df.to_csv("data/california_housing.csv", index=False)

# Separate input and target
X = df.drop("MedHouseVal", axis=1)
y = df["MedHouseVal"]

# Split dataset
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

print("\nTraining records:", len(X_train))
print("Testing records:", len(X_test))

# Initialize models
models = {
    "Multiple Linear Regression": make_pipeline(
        StandardScaler(), LinearRegression()
    ),
    "Decision Tree Regressor": DecisionTreeRegressor(random_state=42),
    "Random Forest Regressor": RandomForestRegressor(
        n_estimators=100, random_state=42, n_jobs=-1
    )
}

# Train and evaluate models
results = []
predictions = {}

for name, model in models.items():
    print("\nTraining:", name)
    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)
    predictions[name] = y_pred

    mae = mean_absolute_error(y_test, y_pred)
    mse = mean_squared_error(y_test, y_pred)
    rmse = mse ** 0.5
    r2 = r2_score(y_test, y_pred)

    results.append({
        "Model": name,
        "MAE": mae,
        "MSE": mse,
        "RMSE": rmse,
        "R2 Score": r2
    })

    print("MAE:", round(mae, 4))
    print("MSE:", round(mse, 4))
    print("RMSE:", round(rmse, 4))
    print("R2 Score:", round(r2, 4))

# Create comparison table
results_df = pd.DataFrame(results)
print("\nFINAL MODEL COMPARISON")
print(results_df.to_string(index=False))
results_df.to_csv("outputs/model_comparison.csv", index=False)

# Correlation heatmap
plt.figure(figsize=(10, 7))
sns.heatmap(df.corr(), annot=True, cmap="coolwarm", fmt=".2f")
plt.title("California Housing Dataset - Correlation Heatmap")
plt.tight_layout()
plt.savefig("outputs/correlation_heatmap.png")
plt.close()

# Actual vs predicted graphs
fig, axes = plt.subplots(1, 3, figsize=(18, 5))

for ax, (name, y_pred) in zip(axes, predictions.items()):
    ax.scatter(y_test, y_pred, alpha=0.3)
    ax.set_title(name)
    ax.set_xlabel("Actual Values")
    ax.set_ylabel("Predicted Values")
    ax.plot(
        [y_test.min(), y_test.max()],
        [y_test.min(), y_test.max()],
        "r--"
    )

plt.suptitle("Actual vs Predicted Values")
plt.tight_layout()
plt.savefig("outputs/actual_vs_predicted.png")
plt.close()

# Model comparison graph
results_df.set_index("Model")[["MAE", "MSE", "RMSE"]].plot(
    kind="bar", figsize=(12, 6)
)
plt.title("Performance Comparison of Machine Learning Models")
plt.ylabel("Error Value")
plt.xticks(rotation=15)
plt.tight_layout()
plt.savefig("outputs/model_comparison.png")
plt.close()

# RMSE comparison graph
plt.figure(figsize=(9, 5))
plt.bar(results_df["Model"], results_df["RMSE"])
plt.title("RMSE Comparison of Machine Learning Models")
plt.xlabel("Machine Learning Models")
plt.ylabel("RMSE")
plt.xticks(rotation=15)
plt.tight_layout()
plt.savefig("outputs/rmse_comparison.png")
plt.close()

# Display best model
best_model = results_df.loc[results_df["R2 Score"].idxmax()]
print("\nBEST PERFORMING MODEL")
print("Model:", best_model["Model"])
print("R2 Score:", round(best_model["R2 Score"], 4))

print("\nAll graphs and results saved in the outputs folder.")
print("\nPROJECT EXECUTION COMPLETED SUCCESSFULLY!")
