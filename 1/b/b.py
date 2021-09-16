import pandas as pd
import numpy as np
import torch
import matplotlib.pyplot as plt
from MultipleLinearRegressionModel import MultipleLinearRegressionModel
from mpl_toolkits import mplot3d

dataset = pd.read_csv(r'/Users/havardtysland/Documents/Dataing 5.  semester/Maskinlæring/1/b/day_length_weight.csv')
#filtered_data = filtered_data = raw_data[~np.isnan(raw_data["y"])]  (Removes NaN)
day = torch.tensor(dataset['# day'].values, dtype=torch.float32).reshape(-1,1)
length = torch.tensor(dataset['length'].values, dtype=torch.float32).reshape(-1,1)
weight = torch.tensor(dataset['weight'].values, dtype=torch.float32).reshape(-1,1)

model = MultipleLinearRegressionModel()

# Optimize: adjust W and b to minimize loss using stochastic gradient descent
optimizer = torch.optim.SGD([model.b, model.W_1, model.W_2], 0.0001)
for epoch in range(150000):
    model.loss(length, weight, day).backward() # Compute loss gradients
    optimizer.step()  # Perform optimization by adjusting W and b,
    optimizer.zero_grad()  # Clear gradients for next step

# Print model variables and loss
print(model.W_1)
print(model.W_2)
print(model.b)
print(model.loss(length, weight, day))
print(f"Estimert alder hvis lengde er 120 og vekt 21: {model.test(120, 21)}")

# Visualize result
fig = plt.figure('Linear regression 3d')
ax = plt.axes(projection='3d')
# Information for making the plot understandable
ax.set_xlabel('Lengde')
ax.set_ylabel('Vekt')
ax.set_zlabel('Alder')

# Plot
ax.scatter(dataset['length'], dataset['weight'], dataset['# day'])

x = torch.tensor([[torch.min(length)], [torch.max(length)]])
y = torch.tensor([[torch.min(weight)], [torch.max(weight)]])
ax.plot(x.flatten(), y.flatten(), model.f(
    x, y).detach().flatten(),  color="red")
ax.legend()
plt.show()



