import torch
import matplotlib.pyplot as plt


# Observed/training input and output. 0 = 1 and 1 = 0
x_train = torch.tensor([[0.0], [1.0]]).reshape(-1, 1)
y_train = torch.tensor([[1.0], [0.0]]).reshape(-1, 1)

class NotOperatorModel:
    def __init__(self):
        self.W = torch.tensor([[0.0]], requires_grad=True)
        self.b = torch.tensor([[0.0]], requires_grad=True)

    def logits(self, x):
        return x @ self.W + self.b
    
    def f(self, x):
        return torch.sigmoid(self.logits(x))

    def loss(self, x, y):
        return torch.nn.functional.binary_cross_entropy_with_logits(self.logits(x), y)


model = NotOperatorModel()

optimizer = torch.optim.SGD([model.b, model.W], 0.1)
for epoch in range(25_000):
    model.loss(x_train, y_train).backward()
    optimizer.step()
    optimizer.zero_grad()

print("W = %s, b=%s, loss = %s" % (model.W, model.b, model.loss(x_train, y_train)))

# Visualize result
plt.figure('Oppgave A')
plt.title('NOT-operator')
plt.table(cellText=[[0, 1], [1, 0]],
          colWidths=[0.1] * 3,
          colLabels=["$x$", "$f(x)$"],
          cellLoc="center",
          loc="lower left")
plt.scatter(x_train, y_train)
plt.xlabel('x')
plt.ylabel('y')
x = torch.arange(0.0, 1.0, 0.001).reshape(-1, 1)
y = model.f(x).detach()
plt.plot(x, y, color="orange")
plt.show()

    