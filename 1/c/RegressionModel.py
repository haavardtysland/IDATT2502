import torch as to
import math

class RegressionModel: 
  def __init__(self): 
    self.W = to.tensor([[0.0]], requires_grad=True)  # requires_grad enables calculation of gradients
    self.b = to.tensor([[0.0]], requires_grad=True)

  # Predictor
  def f(self, x):
    return  20 * 1 / (1 + to.exp(-(x @ self.W + self.b))) + 31  # @ corresponds to matrix multiplication

  def loss(self, x, y):
    return to.nn.functional.mse_loss(self.f(x), y)

  def test (self, x): 
    return 20 * 1 / (1 + math.exp(-(x * self.W.item()))) + 31



