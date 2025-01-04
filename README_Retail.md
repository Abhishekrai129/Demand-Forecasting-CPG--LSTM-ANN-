
# Retail Time Series Analysis and Forecasting

This project demonstrates a complete pipeline for analyzing and forecasting retail sales using time series data. It incorporates feature engineering, model training with hyperparameter optimization, and model evaluation. The methods include classical machine learning, deep learning, and an ensemble of tree-based regressors.

---

## Features

1. **Synthetic Data Generation:**
   - Generates realistic retail sales data with seasonal trends and holiday spikes.
   - Adds time-based features such as day of the week, month, and weekend indicators.

2. **Feature Engineering:**
   - Creates lag features and rolling averages to capture temporal dependencies.

3. **Model Training:**
   - Implements models such as Random Forest, Gradient Boosting, SVR, XGBoost, and LSTM.
   - Optimizes hyperparameters using RandomizedSearchCV and TimeSeriesSplit.

4. **Evaluation:**
   - Evaluates models using RMSE with time series cross-validation.
   - Identifies the best-performing model based on cross-validation results.

5. **Visualization:**
   - Plots synthetic sales data.
   - Compares actual vs predicted sales for the best model.

---

## Libraries Used

- **Data Manipulation and Visualization:**
  - pandas, numpy, matplotlib

- **Machine Learning Models and Utilities:**
  - scikit-learn (Random Forest, Gradient Boosting, SVR, TimeSeriesSplit)
  - xgboost (XGBRegressor)

- **Deep Learning:**
  - tensorflow.keras (LSTM)

- **Warnings Management:**
  - warnings

---

## Workflow

### Step 1: Generate Synthetic Data
- Use sine functions and random noise to simulate seasonal sales data.
- Add holiday-related spikes for realism.

### Step 2: Feature Engineering
- Create lag features and rolling averages.
- Extract temporal features such as day of the week and month.

### Step 3: Train-Test Split
- Perform a time-based split to ensure future data is not used for training.
- Scale features for models sensitive to data range.

### Step 4: Model Training and Hyperparameter Optimization
- Train multiple models (Random Forest, Gradient Boosting, SVR, XGBoost, LSTM).
- Use RandomizedSearchCV with TimeSeriesSplit for hyperparameter tuning.

### Step 5: Model Evaluation
- Evaluate models using RMSE with time series cross-validation.
- Print the best model and its performance.

### Step 6: Visualization
- Plot synthetic sales data.
- Compare actual vs predicted sales for the best model.

---

## File Structure

```
.
├── main.py          # Main script for data analysis and modeling
├── README.md        # Project documentation
├── requirements.txt # List of dependencies
```

---

## How to Use

### Prerequisites

- Python 3.8+
- Required libraries (install via `requirements.txt`):
  ```bash
  pip install -r requirements.txt
  ```

### Running the Code

1. Clone the repository:
   ```bash
   git clone <repository_url>
   cd <repository_name>
   ```

2. Execute the script:
   ```bash
   python main.py
   ```

---

## Results

- Models are evaluated based on RMSE and their ability to capture temporal trends in the sales data.
- The best-performing model is selected and used to predict sales on the test set.
- Visualizations compare actual vs predicted sales.

---

## Future Improvements

1. **Additional Features:**
   - Include external variables like holidays, weather, or promotions.
2. **Advanced Models:**
   - Experiment with transformers and hybrid models.
3. **Scalability:**
   - Optimize code for larger datasets.

---

## Contributing

Contributions are welcome! Feel free to fork this repository, open an issue, or submit a pull request.

---

## License

This project is licensed under the MIT License. See the LICENSE file for details.
