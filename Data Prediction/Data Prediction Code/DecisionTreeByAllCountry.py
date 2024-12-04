import pandas as pd
from sklearn.tree import DecisionTreeRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import numpy as np
import matplotlib.pyplot as plt

root = '/Users/zhujun/LU/CDS504-Business Data Analytics/GroupProject/DataSet/DataPredict/PredictFile/shortcodeTotalMedal_1896-2024.csv'
df = pd.read_csv(root)

df = df.sort_values(by=['country', 'year'])

df = df[~df['year'].isin([2028])]
# 创建滞后特征
df['prev_total'] = df.groupby('country')['Total'].shift(1)
df['prev_2_total'] = df.groupby('country')['Total'].shift(2)
df['avg_last_3'] = df.groupby('country')['Total'].rolling(window=3).mean().shift(1).reset_index(level=0, drop=True)
df = df.dropna()

# 准备特征和目标变量
X = df[['year', 'prev_total', 'prev_2_total', 'avg_last_3']]
y = df['Total']

# 创建决策树回归模型
model = DecisionTreeRegressor(random_state=42)
# 训练模型
model.fit(X, y)
# 获取每个国家的最新数据（即2024年），用于预测2028年
latest_data = df.groupby('country').tail(1)

# 准备2028年的特征
prediction_data = pd.DataFrame({
    'year': [2028] * len(latest_data),
    'prev_total': latest_data['Total'].values,
    'prev_2_total': latest_data['prev_total'].values,
    'avg_last_3': latest_data[['Total', 'prev_total', 'prev_2_total']].mean(axis=1).values
}, index=latest_data['country'])

# 使用训练好的模型预测2028年的奖牌数
predictions = model.predict(prediction_data)

# 将预测结果加入到 DataFrame 中
prediction_data['predicted_total'] = predictions

# 只筛选出指定的国家
selected_countries = ['USA', 'FRA', 'CHN', 'AUS', 'GBR', 'JPN', 'BRA']
selected_predictions = prediction_data[prediction_data.index.isin(selected_countries)]

# 输出这些国家的预测结果
print("\nPredicted medal totals for selected countries in 2028:")
print(selected_predictions[['predicted_total']])
# 评估模型性能
y_pred = model.predict(X)
mse = mean_squared_error(y, y_pred)
rmse = np.sqrt(mse)
mae = mean_absolute_error(y, y_pred)
r2 = r2_score(y, y_pred)

print(f"\nModel performance on historical data:")
print(f"Mean Squared Error (MSE): {mse:.2f}")
print(f"Root Mean Squared Error (RMSE): {rmse:.2f}")
print(f"Mean Absolute Error (MAE): {mae:.2f}")
print(f"Coefficient of Determination (R2): {r2:.2f}")

countries = selected_predictions.index.tolist()
predicted_totals = selected_predictions['predicted_total'].values

plt.figure(figsize=(10, 6))
plt.bar(countries, predicted_totals, color=['blue', 'green', 'red', 'orange', 'purple', 'brown', 'pink'])
plt.xlabel('Country', fontsize=12)
plt.ylabel('Predicted Total Medals', fontsize=12)
plt.title('Predicted Total Medals for 2028 Olympics', fontsize=14)
plt.show()