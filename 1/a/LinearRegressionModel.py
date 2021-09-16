import torch as to

class LinearRegressionModel: 
  def __init__(self): 
    self.W = to.tensor([[0.0]], requires_grad=True)  # requires_grad enables calculation of gradients
    self.b = to.tensor([[0.0]], requires_grad=True)

  # Predictor
  def f(self, weight):
    return weight @ self.W + self.b  # @ corresponds to matrix multiplication

  def loss(self, x, y):
      return to.nn.functional.mse_loss(self.f(x), y)

  def test (self, weight): 
    return weight * self.W.item() + self.b.item()


