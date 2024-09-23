# Time Series Forecasting with Gaussian Process Regression

## Overview

This project applies Gaussian Process Regression (GPR) to solve a time series forecasting problem. The goal is to predict monthly receipt counts for the year 2022 using historical data.

## How Gaussian Process Regression Works

Gaussian Process Regression (GPR) is a powerful, non-parametric Bayesian approach to regression. It provides a flexible framework for modeling complex datasets and is particularly useful when dealing with small datasets. Here's a breakdown of how GPR operates:

### Key Concepts

1. **Gaussian Processes**: 
   - A Gaussian Process is a collection of random variables, any finite number of which have a joint Gaussian distribution. It defines a distribution over functions and is fully specified by its mean function and covariance function (kernel).

2. **Kernel Functions**:
   - Kernels are at the heart of GPR, determining the shape and smoothness of the functions that can be modeled. They measure similarity between data points.
   - **Linear Kernel**: Captures linear relationships in the data.
     $k_{\text{linear}}(x_i, x_j) = \sigma^2 \cdot (x_i \cdot x_j)$
   - **Periodic Kernel**: Captures periodic patterns and seasonality.
     $k_{\text{periodic}}(x_i, x_j) = \sigma^2 \exp\left(-\frac{2\sin^2(\pi |x_i - x_j| / p)}{\ell^2}\right)$

3. **Covariance Matrix**:
   - The covariance matrix $K$ is constructed using the kernel function to describe the relationships between all pairs of training points.

4. **Prediction**:
   - Given training data $X$ and test data $X_\*$, GPR predicts the mean and variance of the function values at $X_\*$.
   - The predictive mean $\mu_\*$ and variance $\sigma_\*$ are given by:
     $\mu_\* = K(X_\*, X)K(X, X)^{-1}y$ and $\sigma_\* = K(X_\*, X_\*) - K(X_\*, X)K(X, X)^{-1}K(X, X_\*)$

5. **Hyperparameter Optimization**:
   - Hyperparameters of the kernel functions are optimized by maximizing the log marginal likelihood:
     $\log p(y|X) = -\frac{1}{2}y^T K^{-1} y - \frac{1}{2}\log|K| - \frac{n}{2}\log 2\pi$
   - This involves finding parameters that best explain the observed data.

### Advantages of GPR

- **Flexibility**: GPR can model a wide range of functions due to its non-parametric nature.
- **Uncertainty Quantification**: Provides confidence intervals for predictions, which is crucial for understanding prediction reliability.
- **Effective with Small Data**: Unlike forecasting models like LSTM and ARIMA, GPR performs well even with limited data.

In this project, GPR was used to forecast monthly receipt counts by leveraging both linear and periodic kernels to capture trends and seasonality in the data. The ability to provide confidence intervals makes it particularly advantageous for decision-making processes where understanding uncertainty is important.

## Project Structure

- `gpr_forecast.py`: Contains the implementation of the GPR model, data preparation, and forecasting functions.
- `app.py`: Flask application to serve the frontend interface.
- `templates/index.html`: HTML template for the web interface.
- `static/styles.css`: Stylesheet for the web interface.
- `data/data_daily.csv`: Input data file containing daily receipt counts.

## Running the Project

### Prerequisites

Ensure you have Python 3.12 and Docker installed on your system.

### Running Manually

1. **Install Dependencies**:
   ```bash
   pip install -r requirements.txt
   ```
   
2. **Run Flask Application**:
  ```bash
  python3 app.py
  ```

3. **Access the Application**:
Open your browser and navigate to `http://localhost:5000` to view the interface and generate forecasts.

### Running with Docker

1. **Build Docker Image**:
   ```bash
   docker build -t my-flask-app .
   ```
2. **Run Docker Container**:
  ```bash
  docker run -p 5000:5000 my-flask-app
  ```
3. **Access the Application**:
Open your browser and navigate to `http://localhost:5000` to view the interface and generate forecasts.


