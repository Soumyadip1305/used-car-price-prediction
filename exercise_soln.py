import sys

import pandas as pd
import torch
from torch import nn

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


accident_free=df['accident']=="None reported"
accident_free=accident_free.astype(int)


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
})


# =========================
# 3. Create X and y
# =========================

# Features
x = torch.column_stack([
    torch.tensor(accident_free,dtype=torch.float32),
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

model = nn.Linear(3, 1)

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
    [1,5, 10000],  # car is 5 years old and runs 10000 miles
    [1,2, 10000],
    [1,5, 20000]
], dtype=torch.float32)

x_data = (x_data - x_mean) / x_std

prediction = model(x_data)

print("\nPredicted price:")
print(prediction * y_std + y_mean)

x_data_accident = torch.tensor([
    [0,5, 10000],  # car is 5 years old and runs 10000 miles
    [0,2, 10000],
    [0,5, 20000]
], dtype=torch.float32)

x_data_accident = (x_data_accident - x_mean) / x_std

prediction = model(x_data_accident)

print("\nPredicted price:")
print(prediction * y_std + y_mean)