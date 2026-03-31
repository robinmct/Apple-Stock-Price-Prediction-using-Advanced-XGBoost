# `Apple Stock Price Prediction using Advanced XGBoost` #

**Dataset:** Apple Stock Prices (2015–2020) – Daily Historical Stock Data

This notebook builds a machine learning pipeline to forecast Apple Inc. (AAPL) daily stock price movements using **XGBoost**, a gradient-boosted tree algorithm well-suited to structured, tabular financial data. Rather than predicting the raw closing price — which is non-stationary and unbounded — the model is trained to predict **next-day percentage returns**, eliminating look-ahead bias and making the target stationary.

### Project Structure
1. **Environment Setup**: Importing all libraries and configuring the plot style.
2. **Data Ingestion and Formatting**: Loading and structuring the raw CSV for time series use.
3. **Feature Engineering**: Constructing financial indicators (MACD, RSI, moving averages, volatility) that summarize past market behavior.
4. **Train/Test Split**: Partitioning data chronologically — training on the past, testing on the future.
5. **Model Training**: Fitting XGBoost with time-aware cross-validation and early stopping.
6. **Evaluation**: Assessing the model against a naive baseline using price error, directional accuracy, and a simulated trading strategy.
7. **Feature Importance Analysis**: Interpreting which features drove the model's predictions.
8. **Monte Carlo Forecasting**: Simulating 1,000 possible 30-day price paths with dynamic, rolling feature reconstruction.
