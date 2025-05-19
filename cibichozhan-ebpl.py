import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

# Load dataset (Replace 'air_quality.csv' with actual dataset)
df = pd.read_csv('air_quality.csv')

# Display basic information
print(df.info())
print(df.describe())

# Handle missing values
df.fillna(df.mean(), inplace=True)

# Select features and target variable
features = ['PM2.5', 'PM10', 'NO2', 'SO2', 'CO', 'Temperature', 'Humidity', 'WindSpeed']
target = 'AQI'

X = df[features]
y = df[target]

# Split data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Normalize features
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Train machine learning model
model = RandomForestRegressor(n_estimators=100, random_state=42)
model.fit(X_train_scaled, y_train)

# Predict air quality levels
y_pred = model.predict(X_test_scaled)

# Evaluate model
mae = mean_absolute_error(y_test, y_pred)
mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

print(f"Mean Absolute Error: {mae}")
print(f"Mean Squared Error: {mse}")
print(f"R-squared Score: {r2}")

# Feature importance visualization
feature_importances = model.feature_importances_
plt.figure(figsize=(10, 5))
sns.barplot(x=features, y=feature_importances)
plt.xlabel("Features")
plt.ylabel("Importance")
plt.title("Feature Importance in Air Quality Prediction")
plt.show()
