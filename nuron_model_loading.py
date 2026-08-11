import sys

import pandas as pd
import torch
from torch import nn

import matplotlib.pyplot as plt

x_mean=torch.load("./model/x_mean.pt")

x_std=torch.load("./model/x_std.pt")

y_mean=torch.load("./model/y_mean.pt")

y_std=torch.load("./model/y_std.pt")


model=nn.Linear(2,1)
model.load_state_dict(torch.load("./model/model.pt",weights_only=True))

model.eval()

x_data = torch.tensor([
    [5, 10000],  # car is 5 years old and runs 10000 miles
    [2, 10000],
    [5, 20000]
], dtype=torch.float32)


with torch.no_grad():
    x_data = (x_data - x_mean) / x_std
    prediction = model(x_data)
    print("\nPredicted price:")
    print(prediction * y_std + y_mean)