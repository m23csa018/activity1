import numpy as np
import matplotlib.pyplot as plt



def relu(x):
    return np.maximum(0, x)

def leaky_relu(x, alpha=0.01):
    return np.where(x > 0, x, alpha * x)

def tanh(x):
    return np.tanh(x)

# Generating input data
random_values = [-3.5, -1.2, 0, 2.8, -4.1, 1.5, -0.7, 3.2, -2.4, 4.6]

# Calculate outputs for each activation function
relu_y = relu(random_values)
leaky_relu_y = leaky_relu(random_values)
tanh_y = tanh(random_values)

# Plotting graphs for each activation function separately

# ReLU
plt.plot(random_values, relu_y, label='ReLU')
plt.title('ReLU')
plt.xlabel('Input')
plt.ylabel('Output')
plt.legend()
plt.tight_layout()
plt.show()

plt.figure(figsize=(6, 4))

# Leaky ReLU
plt.plot(random_values, leaky_relu_y, label='Leaky ReLU')
plt.title('Leaky ReLU')
plt.xlabel('Input')
plt.ylabel('Output')
plt.legend()
plt.tight_layout()
plt.show()

plt.figure(figsize=(6, 4))

# Tanh
plt.plot(random_values, tanh_y, label='Tanh')
plt.title('Tanh')
plt.xlabel('Input')
plt.ylabel('Output')
plt.legend()
plt.tight_layout()
plt.show()
