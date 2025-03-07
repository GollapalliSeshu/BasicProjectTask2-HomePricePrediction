import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
import xgboost as xgb
from sklearn.metrics import mean_squared_error
data = {
    'Income': np.random.randint(30000, 100000, 1000),
    'Schools': np.random.randint(1, 10, 1000),
    'Hospitals': np.random.randint(1, 10, 1000),
    'CrimeRates': np.random.randint(1, 100, 1000),
    'Price': np.random.randint(100000, 500000, 1000)
}

df = pd.DataFrame(data)

X = df[['Income', 'Schools', 'Hospitals', 'CrimeRates']]
y = df['Price']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

model = xgb.XGBRegressor(objective='reg:squarederror', n_estimators=100, learning_rate=0.05, max_depth=5)
model.fit(X_train, y_train)

y_pred = model.predict(X_test)
rmse = np.sqrt(mean_squared_error(y_test, y_pred))

print(f"Root Mean Squared Error: {rmse}")
