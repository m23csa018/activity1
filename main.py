import numpy as np
import matplotlib.pyplot as plt

def sigmoid(x):
    return 1 / (1 + np.exp(-x))

def relu(x):
    return np.maximum(0, x)

def leaky_relu(x, alpha=0.01):
    return np.where(x > 0, x, alpha * x)

def tanh(x):
    return np.tanh(x)

# Generating input data
random_values = [-3.5, -1.2, 0, 2.8, -4.1, 1.5, -0.7, 3.2, -2.4, 4.6]

# Calculate outputs for each activation function
sigmoid_y = sigmoid(random_values)


# Plotting graphs for each activation function separately
plt.figure(figsize=(12, 8))

# Sigmoid
plt.subplot(2, 2, 1)
plt.plot(random_values, sigmoid_y, label='Sigmoid')
plt.title('Sigmoid')
plt.xlabel('Input')
plt.ylabel('Output')
plt.legend()
