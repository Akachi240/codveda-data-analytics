import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.preprocessing import StandardScaler

cols = ['CRIM','ZN','INDUS','CHAS','NOX','RM','AGE','DIS','RAD','TAX','PTRATIO','B','LSTAT','MEDV']
df = pd.read_csv('house_data.csv', header=None, names=cols, sep=r'\s+', engine='python',skiprows=1)
print(df.shape)
print(df.head())
print(df.describe())
print(df.isnull().sum())
X = df.drop('MEDV', axis=1)
Y = df['MEDV']
print(X.shape)
print(Y.shape)
X_train, X_test,Y_train, Y_test = train_test_split(
    X, Y, test_size=0.2, random_state=42
)
print(X_train.shape)
print(X_test.shape)
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)
print("Scaling done!" )
model = LinearRegression()
model.fit(X_train_scaled,Y_train)
print ("Model trained!")
Y_pred = model.predict(X_test_scaled)
mse = mean_squared_error(Y_test, Y_pred)
rmse = np.sqrt(mse)
r2 = r2_score(Y_test, Y_pred)
print(f"R² Score : {r2:.4f}")
print (f"MSE : {mse:.4f}")
print(f"RMSE : {rmse:.4f}")
Y_pred =  model.predict(X_test_scaled)
r2 = r2_score(Y_test, Y_pred)
mse = mean_squared_error(Y_test, Y_pred)
rmse = np.sqrt(mse)
print(f"R² Score : {r2:.4f}")
print(f"MSE : {mse:.4f}")
print (f"RMSE : {rmse:.4f}")
coeff_df = pd.DataFrame({
    'Feature':X.columns, 
    'Coefficient' : model.coef_
})
coeff_df = coeff_df.sort_values('Coefficient', key=abs, ascending=False)
print(coeff_df)
plt.figure()
plt.scatter(Y_test, Y_pred)
min_val = float(Y_test.min())
max_val = float(Y_test.max())
plt.plot([min_val, max_val], [min_val, max_val], 'r--', label='Perfect fit')
plt.xlabel('Actual Price')
plt.ylabel('Predicted Price')
plt.title('Actual va Predicted House Prices')
plt.legend()
plt.savefig('actual_vs_predicted.png')
plt.show()
#Plot 2 - Residuals
plt.figure()
residuals = Y_test - Y_pred
plt.scatter(Y_pred, residuals)
plt.axhline(y=0, color='r', linestyle='--')
plt.xlabel('Predicted Price')
plt.ylabel('Residual Plot')
plt.savefig('residuals.png')
plt.show()

#plot 3 -Feature Coefficient
plt.figure()
coeff_df = pd.DataFrame({'Feature': X.columns, 'Coefficient': model.coef_})
coeff_df = coeff_df.sort_values('Coefficient')
plt.barh(coeff_df['Feature'], coeff_df['Coefficient'])
plt.title('Feature Coefficients')
plt.savefig('coefficients.png')
plt.show()