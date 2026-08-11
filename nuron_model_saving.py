import sys
import os

import pandas as pd
import torch
from torch import nn

import matplotlib.pyplot as plt


# =========================
# 1. Read Dataset
# =========================

df = pd.read_csv(
    r"C:\Users\Soumyadip Saha\DeepLearning\used_cars.csv"
)


# =========================
# 2. Feature Engineering
# =========================

# Age
age = df["model_year"].max() - df["model_year"]


# Mileage
milage = df["milage"]
milage = milage.str.replace(",", "", regex=False)
milage = milage.str.replace(" mi.", "", regex=False)
milage = milage.astype(float)


# Price
price = df["price"]
price = price.str.replace("$", "", regex=False)
price = price.str.replace(",", "", regex=False)
price = price.astype(float)


# =========================
# 3. Create Clean DataFrame
# =========================

data = pd.DataFrame({
    "age": age,
    "milage": milage,
    "price": price
}).dropna()


# =========================
# 4. Create Model Directory
# =========================

if not os.path.isdir("./model"):
    os.mkdir("./model")


# =========================
# 5. Input Features
# =========================

x = torch.column_stack([
    torch.tensor(data["age"].values, dtype=torch.float32),
    torch.tensor(data["milage"].values, dtype=torch.float32)
])


# =========================
# 6. Normalize X
# =========================

x_mean = x.mean(axis=0)
x_std = x.std(axis=0)
torch.save(x_mean,"./model/x_mean.pt")
torch.save(x_std,"./model/x_std.pt")
x = (x - x_mean) / x_std


y = torch.tensor(
    data[["price"]].values,
    dtype=torch.float32
)


# =========================
# 8. Normalize Y
# =========================

y_mean = y.mean()
y_std = y.std()
torch.save(y_mean,"./model/y_mean.pt")
torch.save(y_std,"./model/y_std.pt")
y = (y - y_mean) / y_std


# =========================
# 9. Create Model
# =========================

model = nn.Linear(2, 1)

loss_fn = nn.MSELoss()

optimizer = torch.optim.SGD(
    model.parameters(),
    lr=0.01
)


# =========================
# 10. Training
# =========================

losses = []

for i in range(2500):

    optimizer.zero_grad()
    op = model(x)
    loss = loss_fn(op, y)
    loss.backward()
    optimizer.step()

    losses.append(loss.item())

torch.save(model.state_dict(), "./model/model.pt")
# =========================
# 11. Print Final Results
# =========================

print("Final Loss:", losses[-1])
