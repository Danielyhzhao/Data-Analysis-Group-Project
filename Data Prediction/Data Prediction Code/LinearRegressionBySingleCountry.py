import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
import numpy as np

# 读取数据（注意：请将文件路径改为你自己的路径）
root = '/Users/zhujun/LU/CDS504-Business Data Analytics/GroupProject/DataSet/DataPredict/PredictFile/shortcodeTotalMedal_1896-2024.csv'
df = pd.read_csv(root)

# 只提取美国的数据

country_data = df[df['country'] == 'USA'].copy()
country = 'USA'
country_data = country_data[~country_data['year'].isin([2028])]
# 创建滞后特征和滑动平均特征
country_data['prev_total'] = country_data['Total'].shift(1)
country_data['prev_2_total'] = country_data['Total'].shift(2)
country_data['avg_last_3'] = country_data['Total'].rolling(window=3).mean().shift(1)

# 删除缺失值
country_data = country_data.dropna()

# 准备特征和目标值
X = country_data[['year', 'prev_total', 'prev_2_total', 'avg_last_3']]
y = country_data['Total']

# 创建线性回归模型并训练
model = LinearRegression()
model.fit(X, y)

# 准备用于预测2028年的数据
last_data = country_data.iloc[-1]

prediction_data = pd.DataFrame({
    'year': [2028],
    'prev_total': [last_data['Total']],  # 使用2024年的奖牌总数
    'prev_2_total': [country_data.iloc[-2]['Total']],  # 使用2020年的奖牌总数
    'avg_last_3': [country_data['Total'].tail(3).mean()]  # 最近三届的平均奖牌数
})

# 输出最近三届奥运会的数据
print("\nThe historical data for the last 3 years:")
recent_results = country_data.tail(3)[['year', 'Total']]
print(recent_results.to_string(index=False))

# 预测2028年的奖牌数
predicted_medals = model.predict(prediction_data)[0]
print(f"\nMedal prediction for {country} in 2028:")
print(f"The predicted number of medals: {int(round(predicted_medals))}")

# 评估模型性能
y_pred = model.predict(X)
mse = mean_squared_error(y, y_pred)
rmse = np.sqrt(mse)  # 根均方误差
mae = mean_absolute_error(y, y_pred)
r2 = r2_score(y, y_pred)

print(f"\nModel performance on historical data:")
print(f"Mean Squared Error (MSE): {mse:.2f}")
print(f"Root Mean Squared Error (RMSE): {rmse:.2f}")
print(f"Mean Absolute Error (MAE): {mae:.2f}")
print(f"Coefficient of Determination (R2): {r2:.2f}")