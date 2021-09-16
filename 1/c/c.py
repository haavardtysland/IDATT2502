import pandas as pd
import numpy as np
import torch
import matplotlib.pyplot as plt
from RegressionModel import RegressionModel

dataset = pd.read_csv(r'/Users/havardtysland/Documents/Dataing 5.  semester/Maskinlæring/1/c/day_head_circumference.csv')
#filtered_data = filtered_data = raw_data[~np.isnan(raw_data["y"])]  (Removes NaN)
day = torch.tensor(dataset['# day'].values, dtype=torch.float32).reshape(-1,1)
head = torch.tensor(dataset['head circumference'].values, dtype=torch.float32).reshape(-1,1)

model = RegressionModel()

# Optimize: adjust W and b to minimize loss using stochastic gradient descent
optimizer = torch.optim.SGD([model.W, model.b], 0.00000001)
for epoch in range(10000):
    model.loss(day, head).backward()  # Compute loss gradients
    optimizer.step()  # Perform optimization by adjusting W and b,

    optimizer.zero_grad()  # Clear gradients for next step

# Print model variables and loss
print(model.W)
print(model.b)
print(model.loss(day, head))
print(f"Estimert omkrets hvis alder er 200 dager: {model.test(200)}")

plt.scatter(day, head)
plt.xlabel('Alder')
plt.ylabel('Hodeomkrets')

plt.legend()
plt.show()

