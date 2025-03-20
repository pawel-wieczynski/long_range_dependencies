import numpy as np

def power_law(x: float, a: float, b: float) -> float:
  return a * (x ** b)

def stretched_exponential(x: float, a: float, b: float, c: float) -> float:
  return np.exp(-a * (x ** b) + c)