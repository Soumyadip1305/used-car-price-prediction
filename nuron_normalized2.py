import sys

import pandas as pd
import torch
from torch import nn


# Normalizing the input:



df = pd.read_csv(
    r"C:\Users\Soumyadip Saha\DeepLearning\used_cars.csv"
)


# Age
age = df['model_year'].max() - df['model_year']

# Milage
milage = df['milage']
milage = milage.str.replace(',', '', regex=False)
milage = milage.str.replace(' mi.', '', regex=False)
milage = milage.astype(float)

# Price
price = df['price']
price = price.str.replace('$', '', regex=False)
price = price.str.replace(',', '', regex=False)
price = price.astype(float)

# Remove rows with missing values
data = pd.DataFrame({
    'age': age,
    'milage': milage,
    'price': price
}).dropna()


# =========================
# 3. Create X and y
# =========================

# Features
x = torch.column_stack([
    torch.tensor(age,dtype=torch.float32),
    torch.tensor(milage,dtype=torch.float32)
])
x_mean=x.mean(axis=0)
#print(x.mean(axis=1))
x_std=x.std(axis=0)

x=(x-x_mean)/x_std
# Target
y = torch.tensor(
    data[['price']].values,
    dtype=torch.float32
)


y_mean=y.mean()
y_std=y.std()
# print(y-y_mean)

y=(y-y_mean)/y_std



# =========================
# 5. Create the model
# =========================

model = nn.Linear(2, 1)

loss_fn = nn.MSELoss()

optimizer = torch.optim.SGD(
    model.parameters(),
    lr=0.001
)


# =========================
# 6. Training
# =========================

for i in range(10000):

    # Clear previous gradients
    optimizer.zero_grad()

    op = model(x)
    loss = loss_fn(op, y)

    # Backpropagation
    loss.backward()

    # Update weights
    optimizer.step()

    if i % 100 == 0:
        print(
            f"Iteration: {i}, Loss: {loss.item():.4f}"
        )



x_data = torch.tensor([
    [5, 10000],  # car is 5 years old and runs 10000 miles
    [2, 10000],
    [5, 20000]
], dtype=torch.float32)

x_data = (x_data - x_mean) / x_std

prediction = model(x_data)

print("\nPredicted price:")
print(prediction * y_std + y_mean)