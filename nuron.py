import pandas as pd
import torch
from torch import nn

# =========================
# 1. Read the data
# =========================

df = pd.read_csv(
    r"C:\Users\Soumyadip Saha\DeepLearning\used_cars.csv"
)


# =========================
# 2. Prepare the data
# =========================

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
x = torch.tensor(
    data[['age', 'milage']].values,
    dtype=torch.float32
)

# Target
y = torch.tensor(
    data[['price']].values,
    dtype=torch.float32
)


# =========================
# 4. Scale the data
# =========================

x_mean = x.mean(dim=0)
x_std = x.std(dim=0)

y_mean = y.mean()
y_std = y.std()

x_scaled = (x - x_mean) / x_std
y_scaled = (y - y_mean) / y_std


# =========================
# 5. Create the model
# =========================

model = nn.Linear(2, 1)

loss_fn = nn.MSELoss()

optimizer = torch.optim.SGD(
    model.parameters(),
    lr=0.01
)


# =========================
# 6. Training
# =========================

for i in range(1000):

    # Clear previous gradients
    optimizer.zero_grad()

    # Forward pass
    op = model(x_scaled)

    # Calculate loss
    loss = loss_fn(op, y_scaled)

    # Backpropagation
    loss.backward()

    # Update weights
    optimizer.step()

    if i % 100 == 0:
        print(
            f"Iteration: {i}, Loss: {loss.item():.4f}"
        )


# =========================
# 7. Prediction
# =========================

new_car = torch.tensor(
    [[5, 20000]],
    dtype=torch.float32
)

# Scale new input using training statistics
new_car_scaled = (new_car - x_mean) / x_std

# Prediction
prediction_scaled = model(new_car_scaled)

# Convert prediction back to original price scale
prediction = prediction_scaled * y_std + y_mean

print("\nPredicted price:")
print(prediction.item())