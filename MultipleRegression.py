import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
import matplotlib.pyplot as plt

# 1. CSV 데이터 불러오기
df = pd.read_csv("./sample_data/exchange_data.csv")

# 2. Date 정렬
df['Date'] = pd.to_datetime(df['Date'])
df = df.sort_values(by='Date').reset_index(drop=True)

numeric_cols = ['Price', 'Open', 'High', 'Low']

for col in numeric_cols:
    df[col] = df[col].astype(str).str.replace(',', '').astype(float)

# 3. 다음날 종가 예측을 위한 타깃 생성
df['Next_Price'] = df['Price'].shift(-1)
df = df[:-1]

# # 4. Feature / Target 설정
X = df[['Price', 'Open', 'High', 'Low']]
y = df['Next_Price']

# 5. 데이터 분리 (80% train, 20% test)
train_size = int(len(df) * 0.8)

X_train = X[:train_size]
X_test = X[train_size:]
y_train = y[:train_size]
y_test = y[train_size:]

test_dates = df.loc[X_test.index, 'Date']

# 6. 모델 학습
model = LinearRegression()
model.fit(X_train, y_train)

# 7. 예측
y_pred = model.predict(X_test)
y_pred_series = pd.Series(y_pred, index=y_test.index)

# 8. 오류 측정

percentage_error = ((y_test - y_pred) / y_test) * 100

mse_pct = mean_squared_error(np.zeros(len(percentage_error)), percentage_error)
rmse_pct = np.sqrt(mse_pct)

print("MSE (%) :", mse_pct)
print("RMSE (%):", rmse_pct)

print("MSE (%) :", mse_pct)
print("RMSE (%):", rmse_pct)

# (3) MSE / RMSE 요약 통계 - 포스터 기재용 (평가 지표)
print("\n===== MSE / RMSE Summary (Price 기준) =====")

print("▶ MSE")
print(f"  Min  MSE : {mse.min():.4f}")
print(f"  Max  MSE : {mse.max():.4f}")
print(f"  Mid  MSE : {mse.median():.4f}")
print(f"  Mean MSE : {mse.mean():.4f}")

print("\n▶ RMSE")
print(f"  Min  RMSE : {rmse.min():.4f}")
print(f"  Max  RMSE : {rmse.max():.4f}")
print(f"  Mid  RMSE : {rmse.median():.4f}")
print(f"  Mean RMSE : {rmse.mean():.4f}")

# --- (A) MSE 그래프 ---
plt.figure(figsize=(14, 24))
plt.subplot(3, 1, 1)
plt.plot(test_dates, (y_test - y_pred)**2, linewidth=1)
plt.title("Prediction Error (MSE) by Date", fontsize=15)
plt.xlabel("Date")
plt.ylabel("Squared Error")
plt.grid(True)

# --- (B) RMSE 그래프 ---
plt.subplot(3, 1, 2)
plt.plot(test_dates, np.sqrt((y_test - y_pred)**2), linewidth=1)
plt.title("Prediction Error (RMSE) by Date")
plt.xlabel("Date")
plt.ylabel("Root Squared Error")
plt.grid(True)

plt.subplot(3, 1, 3)
plt.plot(test_dates, y_test, label='Actual Price', linewidth=1.5)
plt.plot(test_dates, y_pred, label='Predicted Price', linewidth=1.5)
plt.title("Actual vs Predicted USD/KRW Price (Test Data)", fontsize=15)
plt.xlabel("Date")
plt.ylabel("Price")
plt.legend()
plt.grid(True)

plt.show()
