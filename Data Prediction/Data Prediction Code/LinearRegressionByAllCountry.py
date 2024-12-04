import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import matplotlib.pyplot as plt
root = '/Users/zhujun/LU/CDS504-Business Data Analytics/GroupProject/DataSet/DataPredict/PredictFile/shortcodeTotalMedal_1896-2024.csv'
df = pd.read_csv(root)

df = df[~df['year'].isin([2028])]
#df = df[df['year'] != 2028]
df_encoded = pd.get_dummies(df, columns=['country'], drop_first=True)
x = df_encoded.drop(['Gold Medal', 'Silver Medal', 'Bronze Medal', 'Total'], axis=1)
# 选择目标列（奖牌总数）
y = df_encoded['Total']
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=42)
# 创建线性回归模型
model = LinearRegression()
# 训练模型
model.fit(x_train, y_train)
# 在测试集上进行预测
y_pred = model.predict(x_test)
# 评估模型
mae = mean_absolute_error(y_test, y_pred)
rmse = mean_squared_error(y_test, y_pred, squared=False)
r2 = r2_score(y_test, y_pred)
print("Mean Absolute Error (MAE):", mae)
print("Root Mean Squared Error (RMSE):", rmse)
print("Coefficient of Determination (R2):", r2)
# 创建 2028 年的预测数据
countries = ['USA', 'FRA', 'CHN', 'AUS', 'GBR', 'JPN', 'BRA']
year_2028 = 2028

# 创建一个 DataFrame，包含 2028 年的国家 One-Hot 编码
data_2028 = pd.DataFrame({
    'year': [year_2028] * len(countries),
    'country': countries
})
# 对国家进行 One-Hot 编码
data_2028_encoded = pd.get_dummies(data_2028, columns=['country'])
# 确保所有国家列与训练集中的 One-Hot 编码列保持一致
for col in x.columns:
    if col not in data_2028_encoded.columns:
        data_2028_encoded[col] = 0  # 如果没有该国家的列，则填充 0
# 按照训练集的列顺序排列
data_2028_encoded = data_2028_encoded[x.columns]
# 使用训练好的模型进行预测
predicted_medals_2028 = model.predict(data_2028_encoded)
# 输出预测结果
for country, medals in zip(countries, predicted_medals_2028):
    print(f"Predicted total medals for {country} in 2028: {medals:.2f}")

# 可视化预测结果
plt.figure(figsize=(8, 5))
plt.bar(countries, predicted_medals_2028, color=['blue', 'green', 'red', 'orange', 'purple', 'brown', 'pink'])
plt.xlabel('Country')
plt.ylabel('Predicted Total Medals')
plt.title('Predicted Total Medals for 2028 Olympics')
plt.show()