import numpy as np
import pandas as pd
import matplotlib

matplotlib.use("Agg")  # Use non-interactive Agg backend
import matplotlib.pyplot as plt
from scipy.optimize import minimize
from scipy.linalg import cholesky, cho_solve
import io
import base64

from typing import Tuple, List


# Define kernel functions used in Gaussian Process Regression
def linear_kernel(X1: np.ndarray, X2: np.ndarray, variance: float) -> np.ndarray:
    """Linear kernel function that computes covariance based on linear relationship."""
    return variance * np.dot(X1, X2.T)


def periodic_kernel(
    X1: np.ndarray, X2: np.ndarray, amplitude: float, length_scale: float, period: float
) -> np.ndarray:
    """Periodic kernel function to capture seasonal patterns in the data."""
    dist_matrix = (
        np.sum(X1**2, 1).reshape(-1, 1) + np.sum(X2**2, 1) - 2 * np.dot(X1, X2.T)
    )
    sin_component = np.sin(np.pi * np.sqrt(dist_matrix) / period)
    return amplitude * np.exp(-2 * (sin_component / length_scale) ** 2)


def combined_kernel(X1: np.ndarray, X2: np.ndarray, params: List[float]) -> np.ndarray:
    """Combines linear and periodic kernels to model both trends and seasonality."""
    return linear_kernel(X1, X2, params[0]) + periodic_kernel(
        X1, X2, params[1], params[2], params[3]
    )


# Define negative log likelihood function for optimization
def negative_log_likelihood(params: List[float], X: np.ndarray, y: np.ndarray) -> float:
    """Calculates the negative log likelihood for Gaussian Process model fitting."""
    K = combined_kernel(X, X, params) + params[4] * np.eye(
        len(X)
    )  # Add noise variance to diagonal
    L = cholesky(
        K, lower=True
    )  # Perform Cholesky decomposition for numerical stability
    alpha = cho_solve((L, True), y)  # Solve for alpha in K*alpha = y
    return (
        0.5 * np.dot(y.T, alpha)
        + np.sum(np.log(np.diag(L)))
        + 0.5 * len(X) * np.log(2 * np.pi)
    )


# Define Gaussian Process Regression model class
class GaussianProcessRegression:
    def __init__(self, kernel):
        self.kernel = kernel
        self.params = None  # To store optimized hyperparameters
        self.X_train = None
        self.y_train = None

    def fit(self, X: np.ndarray, y: np.ndarray) -> None:
        """Fits the Gaussian Process model to training data by optimizing hyperparameters."""
        self.X_train = X
        self.y_train = y

        # Initial guess for hyperparameters: [linear variance, periodic amplitude,
        # periodic length scale, periodic period, noise variance]
        initial_params = [1.0, 1.0, 1.0, 4.0, 0.1]

        # Optimize hyperparameters using L-BFGS-B algorithm with specified bounds
        bounds = [(1e-5, 1e5), (1e-5, 1e5), (1e-5, 1e5), (1, 12), (1e-5, 1e5)]
        result = minimize(
            negative_log_likelihood,
            initial_params,
            args=(X, y),
            method="L-BFGS-B",
            bounds=bounds,
        )
        self.params = result.x

    def predict(self, X_test: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """Predicts mean and variance for test data using the trained model."""
        K = combined_kernel(self.X_train, self.X_train, self.params) + self.params[
            4
        ] * np.eye(len(self.X_train))
        L = cholesky(K, lower=True)
        alpha = cho_solve((L, True), self.y_train)

        K_star = combined_kernel(
            self.X_train, X_test, self.params
        )  # Covariance between training and test data
        f_mean = np.dot(K_star.T, alpha)  # Predicted mean

        v = cho_solve((L, True), K_star)
        f_var = combined_kernel(X_test, X_test, self.params) - np.dot(
            K_star.T, v
        )  # Predicted variance
        return f_mean.flatten(), np.diag(f_var)


def prepare_data() -> Tuple[np.ndarray, ...]:
    """Loads and preprocesses data for training and testing."""
    # Load daily data from CSV file and aggregate to monthly data
    df = pd.read_csv("data/data_daily.csv", parse_dates=["Date"], index_col="Date")
    monthly_data = df.resample("MS").sum()

    # Prepare feature matrix (X) and target vector (y)
    X = np.arange(len(monthly_data)).reshape(-1, 1)
    y = monthly_data["Receipt_Count"].values.reshape(-1, 1)

    # Normalize features and target to have zero mean and unit variance for stability in optimization
    X_mean, X_std = X.mean(), X.std()
    y_mean, y_std = y.mean(), y.std()
    X_normalized = (X - X_mean) / X_std
    y_normalized = (y - y_mean) / y_std

    return (
        X_normalized,
        y_normalized.flatten(),
        X_mean,
        X_std,
        y_mean,
        y_std.flatten(),
        monthly_data,
    )


def generate_forecast() -> Tuple[str, List[dict]]:
    """Generates forecast for future time periods using the trained Gaussian Process model."""

    # Prepare normalized training data
    (
        X_normalized,
        y_normalized,
        X_mean,
        X_std,
        y_mean_flattened,
        y_std_flattened,
        monthly_data,
    ) = prepare_data()

    # Initialize and train Gaussian Process Regression model
    model = GaussianProcessRegression(combined_kernel)
    model.fit(X_normalized.reshape(-1, 1), y_normalized.reshape(-1, 1))

    # Prepare test data for forecasting next 12 months
    X_test = np.arange(len(monthly_data), len(monthly_data) + 12).reshape(-1, 1)
    X_test_normalized = (X_test - X_mean) / X_std

    # Predict normalized mean and variance
    y_pred_normalized, y_var_normalized = model.predict(
        X_test_normalized.reshape(-1, 1)
    )

    # Denormalize predictions to original scale
    y_pred = y_pred_normalized * y_std_flattened + y_mean_flattened
    confidence_interval_upper_bound = (
        y_pred_normalized + 1.96 * np.sqrt(y_var_normalized)
    ) * y_std_flattened + y_mean_flattened
    confidence_interval_lower_bound = (
        y_pred_normalized - 1.96 * np.sqrt(y_var_normalized)
    ) * y_std_flattened + y_mean_flattened

    # Prepare future dates for plotting
    future_dates = pd.date_range(
        start=monthly_data.index[-1] + pd.DateOffset(months=1), periods=12, freq="MS"
    )

    # Create plot of historical data and forecast with confidence intervals
    plt.figure(figsize=(12, 6))
    plt.plot(monthly_data.index, monthly_data["Receipt_Count"], label="Historical Data")
    plt.plot(future_dates, y_pred, label="Monthly Predictions 2022")
    plt.fill_between(
        future_dates,
        confidence_interval_lower_bound,
        confidence_interval_upper_bound,
        alpha=0.2,
    )
    plt.title("Monthly Receipt Count: Historical Data and 2022 Predictions (GPR)")
    plt.xlabel("Date")
    plt.ylabel("Receipt Count")
    plt.legend()

    # Save plot to a bytes buffer for web display
    buf = io.BytesIO()
    plt.savefig(buf, format="png")
    buf.seek(0)
    plot_data = base64.b64encode(buf.getvalue()).decode("utf-8")
    plt.close()

    # Prepare prediction results with confidence intervals
    predictions = []
    for date, prediction, lower_bound, upper_bound in zip(
        future_dates,
        y_pred.flatten(),
        confidence_interval_lower_bound.flatten(),
        confidence_interval_upper_bound.flatten(),
    ):
        predictions.append(
            {
                "month": date.strftime("%b %Y"),
                "prediction": int(prediction),
                "lower_bound": int(lower_bound),
                "upper_bound": int(upper_bound),
            }
        )

    return plot_data, predictions
