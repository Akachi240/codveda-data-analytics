# Codveda Internship - Level 2, Task 1: Regression Analysis
# Dataset: Boston Housing Dataset

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.preprocessing import StandardScaler

# ── 1. LOAD DATASET ──────────────────────────────────────
cols = ['CRIM','ZN','INDUS','CHAS','NOX','RM','AGE',
        'DIS','RAD','TAX','PTRATIO','B','LSTAT','MEDV']

df = pd.read_csv('house_data.csv', header=None, names=cols,
                 sep=r'\s+', engine='python')

print("Shape:", df.shape)
print(df.head())
print("\nDescriptive Stats:\n", df.describe())
print("\nMissing Values:\n", df.isnull().sum())

# ── 2. SPLIT FEATURES & TARGET ───────────────────────────
X = df.drop('MEDV', axis=1)
Y = df['MEDV']

# ── 3. TRAIN/TEST SPLIT ───────────────────────────────────
X_train, X_test, Y_train, Y_test = train_test_split(
    X, Y, test_size=0.2, random_state=42
)
print(f"\nTrain size: {X_train.shape}, Test size: {X_test.shape}")

# ── 4. FEATURE SCALING ────────────────────────────────────
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled  = scaler.transform(X_test)
print("Scaling done!")

# ── 5. TRAIN MODEL ────────────────────────────────────────
model = LinearRegression()
model.fit(X_train_scaled, Y_train)
print("Model trained!")

# ── 6. EVALUATE MODEL ─────────────────────────────────────
Y_pred = model.predict(X_test_scaled)

r2   = r2_score(Y_test, Y_pred)
mse  = mean_squared_error(Y_test, Y_pred)
rmse = np.sqrt(mse)

print(f"\nR² Score : {r2:.4f}")
print(f"MSE      : {mse:.4f}")
print(f"RMSE     : {rmse:.4f}")

# ── 7. COEFFICIENTS TABLE ─────────────────────────────────
coeff_df = pd.DataFrame({
    'Feature'    : X.columns,
    'Coefficient': model.coef_
}).sort_values('Coefficient', key=abs, ascending=False)
print("\nFeature Coefficients:\n", coeff_df.to_string(index=False))

# ── 8. PLOT 1: Actual vs Predicted ────────────────────────
plt.figure(figsize=(7, 5))
plt.scatter(Y_test, Y_pred, alpha=0.7, color='steelblue')
mn, mx = float(Y_test.min()), float(Y_test.max())
plt.plot([mn, mx], [mn, mx], 'r--', label='Perfect Fit')
plt.xlabel('Actual Price')
plt.ylabel('Predicted Price')
plt.title('Actual vs Predicted House Prices')
plt.legend()
plt.tight_layout()
plt.savefig('actual_vs_predicted.png', dpi=150)
plt.show()

# ── 9. PLOT 2: Residuals ──────────────────────────────────
plt.figure(figsize=(7, 5))
residuals = Y_test - Y_pred
plt.scatter(Y_pred, residuals, alpha=0.7, color='coral')
plt.axhline(y=0, color='r', linestyle='--')
plt.xlabel('Predicted Price')
plt.ylabel('Residuals')
plt.title('Residual Plot')
plt.tight_layout()
plt.savefig('residuals.png', dpi=150)
plt.show()

# ── 10. PLOT 3: Feature Coefficients ──────────────────────
plt.figure(figsize=(8, 6))
coeff_sorted = pd.DataFrame({
    'Feature'    : X.columns,
    'Coefficient': model.coef_
}).sort_values('Coefficient')

colors = ['red' if c < 0 else 'steelblue' for c in coeff_sorted['Coefficient']]
plt.barh(coeff_sorted['Feature'], coeff_sorted['Coefficient'], color=colors)
plt.axvline(0, color='black', linewidth=0.8)
plt.title('Feature Coefficients (Linear Regression)')
plt.tight_layout()
plt.savefig('coefficients.png', dpi=150)
plt.show()