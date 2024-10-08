import pandas as pd
import numpy as np
import random
from sklearn.model_selection import TimeSeriesSplit, RandomizedSearchCV
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.svm import SVR
from xgboost import XGBRegressor
from sklearn.metrics import mean_squared_error
from sklearn.preprocessing import StandardScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, LSTM
import matplotlib.pyplot as plt
import warnings

# Suppress warnings for cleaner output
warnings.filterwarnings('ignore')

# Step 1: Create Synthetic Retail Time Series Data
def generate_synthetic_data(num_days=365*2, seed=42):
    np.random.seed(seed)
    rng = pd.date_range(start='2021-01-01', periods=num_days, freq='D')
    sales = np.sin(np.linspace(0, 3 * np.pi, num_days)) * 50 + 200 + np.random.normal(scale=20, size=num_days)
    
    # Adding seasonal spikes for holidays
    for i in range(0, len(sales), 180):  # Increase sales every ~6 months (holidays)
        sales[i:i+10] += random.randint(20, 100)
    
    data = pd.DataFrame({'date_column': rng, 'sales': sales})
    
    # Adding additional time features
    data['day_of_week'] = data['date_column'].dt.dayofweek
    data['month'] = data['date_column'].dt.month
    data['is_weekend'] = data['day_of_week'].apply(lambda x: 1 if x >= 5 else 0)
    
    return data

# Load the synthetic dataset
data = generate_synthetic_data()

# Step 2: Feature Engineering - Create time-based features (e.g., lags, rolling means)
def create_lag_features(df, lags):
    for lag in range(1, lags+1):
        df[f'lag_{lag}'] = df['sales'].shift(lag)
    return df

# Add rolling average and lag features
data['rolling_mean_3'] = data['sales'].rolling(window=3).mean()
data = create_lag_features(data, lags=3)  # Example: Create 3 lag features

# Drop rows with NaN values (due to lagging/rolling)
data = data.dropna()

# Defining X and y
X = data.drop(['sales', 'date_column'], axis=1)  # Drop time column
y = data['sales']

# Time-based Train/Test Split
train_size = int(len(X) * 0.8)  # Use 80% for training and 20% for testing (keeping time order)
X_train, X_test = X[:train_size], X[train_size:]
y_train, y_test = y[:train_size], y[train_size:]

# Step 3: Feature Scaling (Standardization)
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Step 4: Define models and hyperparameter grids
rf = RandomForestRegressor(random_state=42)
gbr = GradientBoostingRegressor(random_state=42)
svr = SVR()
xgb = XGBRegressor(random_state=42, objective='reg:squarederror')

# Hyperparameter Grids (use small grids for demo purposes)
rf_param_grid = {
    'n_estimators': [100, 200],
    'max_depth': [10, 20],
    'min_samples_split': [2, 5]
}
gbr_param_grid = {
    'n_estimators': [100, 200],
    'learning_rate': [0.01, 0.1],
    'max_depth': [3, 5]
}
svr_param_grid = {
    'C': [0.1, 1],
    'kernel': ['rbf', 'linear'],
    'gamma': ['scale']
}
xgb_param_grid = {
    'n_estimators': [100, 200],
    'learning_rate': [0.01, 0.1],
    'max_depth': [3, 5]
}

# Step 5: TimeSeriesSplit for Cross-Validation
tscv = TimeSeriesSplit(n_splits=5)

def random_search_cv(model, param_grid, X_train, y_train, cv):
    random_search = RandomizedSearchCV(
        model, param_grid, n_iter=10, scoring='neg_mean_squared_error', cv=cv, random_state=42, n_jobs=-1
    )
    random_search.fit(X_train, y_train)
    print(f"Best parameters for {model.__class__.__name__}: {random_search.best_params_}")
    return random_search.best_estimator_

# Perform Randomized Search on all models using time series cross-validation
print("Starting RandomizedSearchCV for Random Forest...")
best_rf = random_search_cv(rf, rf_param_grid, X_train_scaled, y_train, tscv)

print("\nStarting RandomizedSearchCV for Gradient Boosting...")
best_gbr = random_search_cv(gbr, gbr_param_grid, X_train_scaled, y_train, tscv)

print("\nStarting RandomizedSearchCV for SVR...")
best_svr = random_search_cv(svr, svr_param_grid, X_train_scaled, y_train, tscv)

print("\nStarting RandomizedSearchCV for XGBoost...")
best_xgb = random_search_cv(xgb, xgb_param_grid, X_train_scaled, y_train, tscv)

# Step 6: Define LSTM (for Time Series) Neural Network
def build_lstm(input_shape):
    model = Sequential()
    model.add(LSTM(50, activation='relu', input_shape=(input_shape[1], input_shape[2])))
    model.add(Dense(1))  # Output layer for regression
    model.compile(optimizer='adam', loss='mean_squared_error')
    return model

# Reshape input for LSTM (samples, timesteps, features)
X_train_lstm = X_train_scaled.reshape((X_train_scaled.shape[0], 1, X_train_scaled.shape[1]))
X_test_lstm = X_test_scaled.reshape((X_test_scaled.shape[0], 1, X_test_scaled.shape[1]))

# Initialize LSTM model
lstm = build_lstm(X_train_lstm.shape)

# Train LSTM model
print("\nTraining LSTM model...")
history = lstm.fit(
    X_train_lstm, y_train,
    epochs=50,
    batch_size=32,
    validation_split=0.2,
    verbose=0  # Set to 1 if you want to see training progress
)

# Step 7: Evaluate Models using TimeSeriesSplit CV
def time_series_cross_val(model, X_train, y_train, cv, is_lstm=False):
    scores = []
    if is_lstm:
        for train_idx, val_idx in cv.split(X_train):
            # Reshape for LSTM
            X_tr = X_train[train_idx].reshape((X_train[train_idx].shape[0], 1, X_train[train_idx].shape[1]))
            X_val = X_train[val_idx].reshape((X_train[val_idx].shape[0], 1, X_train[val_idx].shape[1]))
            
            # Build a new LSTM model for each fold to prevent weight sharing
            model_fold = build_lstm(X_tr.shape)
            model_fold.fit(X_tr, y_train.iloc[train_idx], epochs=50, batch_size=32, verbose=0)
            y_pred = model_fold.predict(X_val).flatten()
            mse = mean_squared_error(y_train.iloc[val_idx], y_pred)
            scores.append(np.sqrt(mse))  # RMSE
    else:
        for train_idx, val_idx in cv.split(X_train):
            model.fit(X_train[train_idx], y_train.iloc[train_idx])
            y_pred = model.predict(X_train[val_idx])
            mse = mean_squared_error(y_train.iloc[val_idx], y_pred)
            scores.append(np.sqrt(mse))  # RMSE
    return np.mean(scores), np.std(scores)

# Evaluate Random Forest, Gradient Boosting, SVR, XGBoost, and LSTM
print("\nEvaluating models with cross-validation...")

rf_cv_rmse, rf_cv_std = time_series_cross_val(best_rf, X_train_scaled, y_train, tscv)
print(f"Random Forest - Cross-validated RMSE: {rf_cv_rmse:.4f} ± {rf_cv_std:.4f}")

gbr_cv_rmse, gbr_cv_std = time_series_cross_val(best_gbr, X_train_scaled, y_train, tscv)
print(f"Gradient Boosting - Cross-validated RMSE: {gbr_cv_rmse:.4f} ± {gbr_cv_std:.4f}")

svr_cv_rmse, svr_cv_std = time_series_cross_val(best_svr, X_train_scaled, y_train, tscv)
print(f"SVR - Cross-validated RMSE: {svr_cv_rmse:.4f} ± {svr_cv_std:.4f}")

xgb_cv_rmse, xgb_cv_std = time_series_cross_val(best_xgb, X_train_scaled, y_train, tscv)
print(f"XGBoost - Cross-validated RMSE: {xgb_cv_rmse:.4f} ± {xgb_cv_std:.4f}")

lstm_cv_rmse, lstm_cv_std = time_series_cross_val(lstm, X_train_scaled, y_train, tscv, is_lstm=True)
print(f"LSTM - Cross-validated RMSE: {lstm_cv_rmse:.4f} ± {lstm_cv_std:.4f}")

# Step 8: Print Results
best_models = {
    'Random Forest': rf_cv_rmse,
    'Gradient Boosting': gbr_cv_rmse,
    'SVR': svr_cv_rmse,
    'XGBoost': xgb_cv_rmse,
    'LSTM': lstm_cv_rmse
}

best_model = min(best_models, key=best_models.get)
print(f"\nBest model: {best_model} with Cross-validated RMSE: {best_models[best_model]:.4f}")

# Optional: Plotting the synthetic data for visualization
plt.figure(figsize=(12,6))
plt.plot(data['date_column'], data['sales'], label='Sales')
plt.title('Synthetic Sales Data Over Time')
plt.xlabel('Date')
plt.ylabel('Sales')
plt.legend()
plt.show()

# Optional: Plot Actual vs Predicted for the best model on the test set
if best_model != 'LSTM':
    # For tree-based models and SVR
    best_estimator = globals()[f"best_{best_model.lower().replace(' ', '_')}"]
    y_pred_test = best_estimator.predict(X_test_scaled)
else:
    # For LSTM
    y_pred_test = lstm.predict(X_test_lstm).flatten()

# Plot Actual vs Predicted Sales
plt.figure(figsize=(12,6))
plt.plot(data['date_column'][train_size:], y_test, label='Actual Sales')
plt.plot(data['date_column'][train_size:], y_pred_test, label='Predicted Sales', alpha=0.7)
plt.title(f'Actual vs Predicted Sales using {best_model}')
plt.xlabel('Date')
plt.ylabel('Sales')
plt.legend()
plt.show()
