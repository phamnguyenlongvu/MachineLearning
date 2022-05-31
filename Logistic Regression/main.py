from random import shuffle
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

def sigmoid(Z): 
    return 1/(1 + np.exp(-Z))

def cost(h, y):
    return (-y * np.log(h) - (1 - y) * np.log(1 - h)).mean()

def gradient(X, h, y):
    return np.dot(X.T, (h - y)) / y.shape[0]

def logistic_regression(X, y, theta, alpha, iters):
    cost_array = np.zeros(iters)
    for i in range(iters):
        h = sigmoid(np.dot(X, theta))
        cost_num = cost(h, y)
        cost_array[i] = cost_num
        gradient_val = gradient(X, h, y)
        theta = theta - (gradient_val * alpha)
    return theta, cost_array

# import data
data = pd.read_csv(r'D:\Machine Learning\Logistic Regression\irisdata.csv')
shuffled = data.sample(frac=1)
# print(shuffled)
# print(shuffled.shape)

# extract data
X = shuffled[['length', 'width']]
y = shuffled['Type']

# Add one column to allow vectorized calculations
X = np.concatenate((np.ones((X.shape[0], 1)), X), axis=1)
# print(X)
# split train, test data
X_train = X[:95]
y_train = y[:95]
X_test = X[95:]
y_test = y[95:]


# Initial theta values
theta = np.zeros(X_train.shape[1])
# print(theta)
# define hyperparameters
alpha = 0.1
iterations = 1000

# Starting values
h = sigmoid(np.dot(X_train, theta))
# With 2 feature and 1 bias
print("Initial cost value for theta values {0} is: {1}".format(theta, cost(h, y_train)))

# Run logistic regression
theta, cost_arr = logistic_regression(X_train, y_train, theta, alpha, iterations)

print(theta)

predict_arr = []
for x in X_test:
    h = sigmoid(np.dot(x, theta))
    if h > 0.5:
        predict_arr.append(1)
    elif h < 0.5:
        predict_arr.append(0)
print("Predict label: {0}".format(predict_arr))
print("Actual label:  {0}".format(y_test))