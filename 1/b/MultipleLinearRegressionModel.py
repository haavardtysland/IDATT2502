import torch as to
class MultipleLinearRegressionModel: 
  def __init__(self): 
    self.W_1 = to.tensor([[0.0]], requires_grad=True)
    self.W_2 = to.tensor([[0.0]], requires_grad=True)
    self.b = to.tensor([[0.0]], requires_grad=True)

  def f(self, length, weight): 
    return (length @ self.W_1) + (weight @ self.W_2) + self.b

  def loss(self, x_1, x_2, y):
    return to.nn.functional.mse_loss(self.f(x_1, x_2), y)

  def test (self, length, weight): 
    return length * self.W_1.item() + weight * self.W_2.item() + self.b.item()