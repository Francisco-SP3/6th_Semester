# Obtain the linear regression of a given 50 point dataset by cost function minimization

import numpy as np
import matplotlib.pyplot as plt

# Generate a dataset
np.random.seed(100)
X = 2 * np.random.rand(50, 1)
y = 4 + 3 * X + np.random.randn(50, 1)

# Plot the dataset
plt.scatter(X, y)
plt.xlabel('X')
plt.ylabel('y')
plt.title('Dataset')
plt.show()

# Add the bias term
X_b = np.c_[np.ones((50, 1)), X]

# Define the learning rate and number of iterations
lr = 0.1
n_iterations = 1000

# Initialize the weights
theta = np.random.randn(2, 1)

# Define the cost function
def cost_function(X, y, theta):
    m = len(y)
    predictions = X.dot(theta)
    cost = (1/2*m) * np.sum(np.square(predictions - y))
    return cost

# Define the gradient descent function
def gradient_descent(X, y, theta, lr, n_iterations):
    m = len(y)
    cost_history = np.zeros(n_iterations)
    for i in range(n_iterations):
        predictions = X.dot(theta)
        errors = np.subtract(predictions, y)
        sum_delta = (1/m) * X.T.dot(errors)
        theta = theta - lr * sum_delta
        cost_history[i] = cost_function(X, y, theta)
    return theta, cost_history

# Run the gradient descent algorithm
theta, cost_history = gradient_descent(X_b, y, theta, lr, n_iterations)

# Print the final weights
print('Theta0:', theta[0][0])
print('Theta1:', theta[1][0])

# Plot the cost function
plt.plot(range(1, n_iterations + 1), cost_history, color='blue')
plt.grid()
plt.xlabel('Number of iterations')
plt.ylabel('Cost (J)')
plt.title('Cost function')
plt.show()

# Plot the linear regression
plt.scatter(X, y)
plt.plot(X, X_b.dot(theta), color='red')

# Show the plot
plt.xlabel('X')
plt.ylabel('y')
plt.title('Linear Regression')
plt.show()

