import pandas as pd
import numpy as np
import torch
import matplotlib.pyplot as plt
from LinearRegressionModel import LinearRegressionModel

dataset = pd.read_csv(r'/Users/havardtysland/Documents/Dataing 5.  semester/Maskinlæring/1/a/length_weight.csv')
x = torch.tensor(dataset['# length'].values, dtype=torch.float32).reshape(-1,1)
y = torch.tensor(dataset['weight'].values, dtype=torch.float32).reshape(-1,1)

model = LinearRegressionModel()

#Optimize: adjust W and b to minimize loss using stochastic gradient descent (SGD)
optimizer = torch.optim.SGD([model.W, model.b], 0.0001)
for epoch in range(500_00):
    model.loss(x,y).backward() #Compute loss gradients
    optimizer.step() #Perform optimization by adjusting W and b

    optimizer.zero_grad() #Clear gradients for next stop

# Print model variables and loss
print(model.W)
print(model.b)
print(model.loss(x, y))
print(f"Estimert vekt hvis lengde er 120: {model.test(120)}")

plt.plot(x, y, 'o')
plt.xlabel('Lengde')
plt.ylabel('Vekt')
x = torch.tensor([[torch.min(x)], [torch.max(x)]])  # x = [[1], [6]]]
plt.plot(x, model.f(x).detach())
plt.legend()
plt.show()
