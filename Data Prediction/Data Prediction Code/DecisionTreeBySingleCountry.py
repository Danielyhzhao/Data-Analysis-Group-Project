import pandas as pd
from sklearn.tree import DecisionTreeRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score
import matplotlib.pyplot as plt

root = '/Users/zhujun/LU/CDS504-Business Data Analytics/GroupProject/DataSet/DataPredict/PredictFile/shortcodeTotalMedal_1896-2024.csv'
df = pd.read_csv(root)

# read the given countr data
country_data = df[df['country'] == 'USA'].copy()
country = 'USA'
country_data = country_data[~country_data['year'].isin([2028])]
# create a new dataframe for the given country data
country_data['prev_total'] = country_data['Total'].shift(1)
country_data['prev_2_total'] = country_data['Total'].shift(2)
country_data['avg_last_3'] = country_data['Total'].rolling(window=3).mean().shift(1)

# remove the missing values
country_data = country_data.dropna()

# prepare the data for training
X = country_data[['year', 'prev_total', 'prev_2_total', 'avg_last_3']]
y = country_data['Total']

# train test split
model = DecisionTreeRegressor(max_depth=5, random_state=42)
model.fit(X, y)

# predict the 2028 medal count
last_data = country_data.iloc[-1]

prediction_data = pd.DataFrame({
    'year': [2028],
    'prev_total': [last_data['Total']],  # 2024
    'prev_2_total': [country_data.iloc[-2]['Total']],  # 2020
    'avg_last_3': [country_data['Total'].tail(3).mean()]  # last 3 years mean
})

# output the historical data for the last 3 years
print("\nThe historical data for the last 3 years:")
recent_results = country_data.tail(3)[['year', 'Total']]
print(recent_results.to_string(index=False))

# predict the medal count for 2028
predicted_medals = model.predict(prediction_data)[0]

print(f"\nMedal prediction for {country} in 2028:")
print(f"The predicted number of medals: {int(round(predicted_medals))} ")

# output the model performance metrics
y_pred = model.predict(X)
mse = mean_squared_error(y, y_pred)
r2 = r2_score(y, y_pred)
print(f"Mean Squared Error (MSE): {mse:.2f}")
print(f"Coefficient of Determination (R2): {r2:.2f}")


