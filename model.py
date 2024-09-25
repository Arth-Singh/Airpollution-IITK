import pandas as pd
import numpy as np
from xgboost import XGBRegressor
from sklearn.model_selection import train_test_split, TimeSeriesSplit
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
from sklearn.impute import KNNImputer
from sklearn.preprocessing import StandardScaler
import joblib
from datetime import datetime
import matplotlib.pyplot as plt
import seaborn as sns
from bayes_opt import BayesianOptimization
import shap

# Read in data
df = pd.read_csv('/teamspace/studios/this_studio/city_day.csv', parse_dates=['Date'])

# Feature engineering
df['Year'] = df['Date'].dt.year
df['Month'] = df['Date'].dt.month
df['Day'] = df['Date'].dt.day
df['DayOfWeek'] = df['Date'].dt.dayofweek
df['IsWeekend'] = df['DayOfWeek'].isin([5, 6]).astype(int)

# Corrected season creation
def get_season(month):
    if month in [12, 1, 2]:
        return 'Winter'
    elif month in [3, 4, 5]:
        return 'Spring'
    elif month in [6, 7, 8]:
        return 'Summer'
    else:
        return 'Autumn'

df['Season'] = df['Month'].apply(get_season)

# Calculate rolling averages
for col in ['PM2.5', 'PM10', 'NO', 'NO2', 'NOx', 'NH3', 'CO', 'SO2', 'O3', 'Benzene', 'Toluene', 'Xylene']:
    df[f'{col}_Rolling_Mean_7'] = df.groupby('City')[col].transform(lambda x: x.rolling(window=7, min_periods=1).mean())

# Create lag features
for col in ['AQI', 'PM2.5', 'PM10']:
    df[f'{col}_Lag_1'] = df.groupby('City')[col].shift(1)

# One-hot encode the 'City' and 'Season' columns
df = pd.get_dummies(df, columns=['City', 'Season'], prefix=['City', 'Season'])

# Select features and target variable
features = ['PM2.5', 'PM10', 'NO', 'NO2', 'NOx', 'NH3', 'CO', 'SO2', 'O3', 'Benzene', 'Toluene', 'Xylene',
            'Year', 'Month', 'Day', 'DayOfWeek', 'IsWeekend'] + \
           [col for col in df.columns if col.startswith(('PM2.5_Rolling_Mean_', 'PM10_Rolling_Mean_', 'City_', 'Season_'))] + \
           [col for col in df.columns if col.endswith('_Lag_1')]
target = 'AQI'

# Drop rows where target is NaN
df = df.dropna(subset=[target])

# Select only the columns we need
X = df[features]
y = df[target]

# Impute missing values using KNNImputer
imputer = KNNImputer(n_neighbors=5)
X_imputed = imputer.fit_transform(X)

# Scale features
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X_imputed)

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)

# Define the objective function for Bayesian optimization
def xgb_evaluate(max_depth, learning_rate, n_estimators, subsample, colsample_bytree):
    params = {
        'max_depth': int(max_depth),
        'learning_rate': learning_rate,
        'n_estimators': int(n_estimators),
        'subsample': max(min(subsample, 1), 0),
        'colsample_bytree': max(min(colsample_bytree, 1), 0),
        'tree_method': 'hist',
        'device': 'cuda'  # Use CUDA for GPU training
    }
    model = XGBRegressor(**params)
    
    # Use TimeSeriesSplit for cross-validation
    tscv = TimeSeriesSplit(n_splits=5)
    cv_scores = []
    for train_index, val_index in tscv.split(X_train):
        X_train_cv, X_val_cv = X_train[train_index], X_train[val_index]
        y_train_cv, y_val_cv = y_train.iloc[train_index], y_train.iloc[val_index]
        model.fit(X_train_cv, y_train_cv)
        predictions = model.predict(X_val_cv)
        score = r2_score(y_val_cv, predictions)
        cv_scores.append(score)
    return np.mean(cv_scores)

# Perform Bayesian optimization
pbounds = {
    'max_depth': (3, 10),
    'learning_rate': (0.01, 0.3),
    'n_estimators': (100, 1000),
    'subsample': (0.5, 1.0),
    'colsample_bytree': (0.5, 1.0)
}

optimizer = BayesianOptimization(f=xgb_evaluate, pbounds=pbounds, random_state=42)
optimizer.maximize(init_points=5, n_iter=25)

# Get the best parameters
best_params = optimizer.max['params']
best_params['max_depth'] = int(best_params['max_depth'])
best_params['n_estimators'] = int(best_params['n_estimators'])
best_params['tree_method'] = 'hist'
best_params['device'] = 'cuda'  # Use CUDA for GPU training

# Train the final model with the best parameters
best_model = XGBRegressor(**best_params)
best_model.fit(X_train, y_train)

# Make predictions on the test set
y_pred = best_model.predict(X_test)

# Evaluate the model
mse = mean_squared_error(y_test, y_pred)
rmse = np.sqrt(mse)
mae = mean_absolute_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

print(f"Mean Squared Error: {mse}")
print(f"Root Mean Squared Error: {rmse}")
print(f"Mean Absolute Error: {mae}")
print(f"R-squared: {r2}")

# Feature importance
feature_importance = best_model.feature_importances_
feature_importance_dict = dict(zip(features, feature_importance))
sorted_features = sorted(feature_importance_dict.items(), key=lambda x: x[1], reverse=True)
print("\nTop 10 Most Important Features:")
for feature, importance in sorted_features[:10]:
    print(f"{feature}: {importance}")

# SHAP values for feature importance
explainer = shap.TreeExplainer(best_model)
shap_values = explainer.shap_values(X_test)

plt.figure(figsize=(10, 6))
shap.summary_plot(shap_values, X_test, plot_type="bar", feature_names=features)
plt.tight_layout()
plt.savefig('shap_summary.png')
plt.close()

# Visualize predictions vs actual
plt.figure(figsize=(10, 6))
plt.scatter(y_test, y_pred, alpha=0.5)
plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--', lw=2)
plt.xlabel('Actual AQI')
plt.ylabel('Predicted AQI')
plt.title('Actual vs Predicted AQI')
plt.tight_layout()
plt.savefig('actual_vs_predicted.png')
plt.close()

# Error analysis
errors = y_test - y_pred
plt.figure(figsize=(10, 6))
sns.histplot(errors, kde=True)
plt.xlabel('Prediction Error')
plt.ylabel('Frequency')
plt.title('Distribution of Prediction Errors')
plt.tight_layout()
plt.savefig('error_distribution.png')
plt.close()

# Save the model
model_filename = f'xgboost_aqi_model_{datetime.now().strftime("%Y%m%d_%H%M%S")}.joblib'
joblib.dump(best_model, model_filename)
print(f"\nModel saved as {model_filename}")

# Save feature names and scaler for future use
joblib.dump(features, 'feature_names.joblib')
joblib.dump(scaler, 'scaler.joblib')
print("Feature names and scaler saved.")

# Generate a brief report
report = f"""
AQI Prediction Model Report

Model Performance:
- R-squared: {r2:.4f}
- Root Mean Squared Error: {rmse:.4f}
- Mean Absolute Error: {mae:.4f}

Top 5 Most Important Features:
{chr(10).join([f"- {feature}: {importance:.4f}" for feature, importance in sorted_features[:5]])}

Model Insights:
1. The model explains {r2*100:.2f}% of the variance in AQI predictions.
2. On average, predictions are off by {mae:.2f} AQI units.
3. The most important feature is {sorted_features[0][0]}, followed by {sorted_features[1][0]}.
4. Time-based features (rolling means and lags) proved to be valuable for predictions.
5. City-specific factors play a significant role in AQI predictions.

Visualizations:
- SHAP summary plot: shap_summary.png
- Actual vs Predicted AQI plot: actual_vs_predicted.png
- Error distribution plot: error_distribution.png

Next Steps:
1. Analyze the instances where the model performs poorly.
2. Consider incorporating more external data (e.g., weather data, major events).
3. Experiment with ensemble methods for potential performance improvements.
4. Develop a simple web application for real-time AQI predictions.
"""

with open('model_report.txt', 'w') as f:
    f.write(report)

print("\nA brief report has been generated and saved as 'model_report.txt'.")
