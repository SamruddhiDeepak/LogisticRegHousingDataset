# Import libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

# Step 1: Load the dataset
df = pd.read_csv("Housing.csv") 

# Step 2: Preprocess the data
# Convert categorical variables to dummy/indicator variables
df = pd.get_dummies(df, drop_first=True)

# Print the first few rows
print("Data preview:\n", df.head())


print("\n--- Simple Linear Regression: Area vs Price ---")
X_simple = df[['area']]           # Feature: area
y = df['price']                   # Target: price

# Split into train and test sets
X_train_s, X_test_s, y_train_s, y_test_s = train_test_split(X_simple, y, test_size=0.2, random_state=42)

# Fit the model
model_simple = LinearRegression()
model_simple.fit(X_train_s, y_train_s)

# Predict and evaluate
y_pred_s = model_simple.predict(X_test_s)
print(f"MAE: {mean_absolute_error(y_test_s, y_pred_s):.2f}")
print(f"MSE: {mean_squared_error(y_test_s, y_pred_s):.2f}")
print(f"R² Score: {r2_score(y_test_s, y_pred_s):.2f}")
print("Intercept:", model_simple.intercept_)
print("Coefficient for area:", model_simple.coef_[0])

# Plot regression line
plt.figure(figsize=(8,5))
plt.scatter(X_test_s, y_test_s, color='blue', label='Actual Price')
plt.plot(X_test_s, y_pred_s, color='red', linewidth=2, label='Predicted Line')
plt.title("Simple Linear Regression: Area vs Price")
plt.xlabel("Area (sq ft)")
plt.ylabel("Price")
plt.legend()
plt.grid(True)
plt.show()

# ----------- MULTIPLE LINEAR REGRESSION (using all features) -----------

print("\n--- Multiple Linear Regression: All Features ---")
X_multi = df.drop("price", axis=1)
y = df["price"]

# Split into train and test sets
X_train_m, X_test_m, y_train_m, y_test_m = train_test_split(X_multi, y, test_size=0.2, random_state=42)

# Fit the model
model_multi = LinearRegression()
model_multi.fit(X_train_m, y_train_m)

# Predict and evaluate
y_pred_m = model_multi.predict(X_test_m)
print(f"MAE: {mean_absolute_error(y_test_m, y_pred_m):.2f}")
print(f"MSE: {mean_squared_error(y_test_m, y_pred_m):.2f}")
print(f"R² Score: {r2_score(y_test_m, y_pred_m):.2f}")
print("Intercept:", model_multi.intercept_)

# Display coefficients with feature names
coeff_df = pd.DataFrame({
    "Feature": X_multi.columns,
    "Coefficient": model_multi.coef_
})
print("\nFeature Coefficients:\n", coeff_df)
