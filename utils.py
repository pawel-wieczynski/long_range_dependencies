import numpy as np

def power_law(x: float, a: float, b: float) -> float:
  return a * (x ** b)

def stretched_exponential(x: float, a: float, b: float, c: float) -> float:
  return np.exp(-a * (x ** b) + c)

def calculate_arsr(y_obs: np.array, y_pred: np.array) -> float:
    return np.mean((y_obs - y_pred)**2 / y_obs**2)

def calculate_sslr(y_obs: np.array, y_pred: np.array, num_params: int = 0) -> float:
    # Filter out non-positive values that would cause log errors
    mask = (y_obs > 0) & (y_pred > 0)
    y_obs_filtered = y_obs[mask]
    y_pred_filtered = y_pred[mask]

    N = len(y_obs_filtered)
    if N <= num_params:
        return float('inf')  # Avoid division by zero
        
    # Sum of squared logarithmic residuals
    log_residuals = np.sum((np.log(y_obs_filtered) - np.log(y_pred_filtered))**2)
    
    # Divide by degrees of freedom (N - number of parameters)
    return log_residuals / (N - num_params)
