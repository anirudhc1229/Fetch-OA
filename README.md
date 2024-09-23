# Time Series Forecasting with Gaussian Process Regression

## Overview

This project applies Gaussian Process Regression (GPR) to solve a time series forecasting problem. The goal is to predict monthly receipt counts for the year 2022 using historical data. GPR is chosen for its ability to model complex patterns and provide confidence intervals, which are particularly valuable when training data is scarce.

### Why Gaussian Process Regression?

Gaussian Process Regression is a non-parametric, Bayesian approach to regression that is particularly well-suited for small datasets. Unlike traditional methods such as ARIMA or LSTM, GPR can effectively capture underlying trends and seasonal patterns without requiring a large amount of data. Additionally, GPR provides confidence intervals for its predictions, offering insights into the uncertainty of the forecasts.

### Kernel Selection and Hyperparameter Training

The model uses a combination of linear and periodic kernels:
- **Linear Kernel**: Captures linear trends in the data.
- **Periodic Kernel**: Models seasonal patterns with a specified period.

Hyperparameters are optimized using the negative log-likelihood function, which measures the fit of the model to the data. This optimization is performed using the L-BFGS-B algorithm, which efficiently handles bound constraints on parameters.

## Project Structure

- `GaussianProcessRegression.py`: Contains the implementation of the GPR model, data preparation, and forecasting functions.
- `app.py`: Flask application to serve the frontend interface.
- `templates/index.html`: HTML template for the web interface.
- `static/styles.css`: Stylesheet for the web interface.
- `data/data_daily.csv`: Input data file containing daily receipt counts.

## Running the Project

### Prerequisites

Ensure you have Python 3.12 and Docker installed on your system. Also, install necessary Python packages if running manually.

### Running Manually

1. **Install Dependencies**:
   ```bash
   pip install -r requirements.txt
   
2. **Run Flask Application**:
  ```bash
  python3 app.py

3. **Access the Application**:
Open your browser and navigate to `http://localhost:5000` to view the interface and generate forecasts.

### Running with Docker

1. **Build Docker Image**:
   ```bash
   docker build -t my-flask-app .
   
2. **Run Docker Container**:
  ```bash
  docker run -p 5000:5000 my-flask-app

3. **Access the Application**:
Open your browser and navigate to `http://localhost:5000` to view the interface and generate forecasts.


