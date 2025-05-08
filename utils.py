import numpy as np

def power_law(x: float, a: float, b: float) -> float:
  return a * (x ** b)

def stretched_exponential(x: float, a: float, b: float, c: float) -> float:
  return np.exp(-a * (x ** b) + c)

def calculate_arsr(y_obs: np.array, y_pred: np.array) -> float:
    return np.mean((y_obs - y_pred)**2 / y_obs**2)
