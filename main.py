import numpy as np
import matplotlib.pyplot as plt

# The Threshold y = {-1, 1}

# Postive Points = (0, 0), (-1, 0), (1, 1)
# Negative Points = (0, 1), (1, 0), (-1, -1)

# Weight Formula =  W_new = W_old + (learning_rate * y * x)
# Bias Update = B_new = B_old + (learning_rate * y)

# g(x) = W_Transpose * X + Bias
# y_pred = activation_function(g(x))

# Input = x
# Weight = W
# Bias = B
# Activation Function = f(x)

class Perceptron:
    def __init__(self, num_input, learning_rate=1):
        self.learning_rate = learning_rate
        self.weights = np.random.rand(num_input + 1)

    # Single Layer g(x) 
    def score(self, X):
        X = np.asarray(X)
        if X.ndim == 1:
            return X @ self.weights[1:] + self.weights[0] # Dot product of the input and the weights
        else:
            return X @ self.weights[1:] + self.weights[0] # Dot product of the input and the weights

    # Activation function
    def activation_function(self, x):
        if x >= 0:
            return 1
        else:
            return -1
    # Prediction
    def predict(self, X):
        gx = self.score(X) # Calculate the score
        if np.ndim(gx) == 0: # If the score is a scalar
            return self.activation_function(gx) # Then return the activation function
        return np.where(gx >= 0, 1, -1) # Otherwise return 1 if the score is greater than 0, otherwise return -1
    
    # Update weights for one sample
    def update_weights(self, x, y):
        z = self.score(x) # Calculate the score
        if y * z <= 0: # If the sample is misclassified
            self.weights[1:] += (self.learning_rate * y * x)  # Updating the weights
            self.weights[0] += (self.learning_rate * y) # Updating the bias
            return True # Return True if the sample is misclassified
        return False # Return False if the sample is correctly classified
    
    # Fit the model
    def fit(self, X, y, epochs=50):
        for _ in range(epochs):
            mistakes = 0
            for xi, yi in zip(X, y):
                mistakes += self.update_weights(xi, yi)
            if mistakes == 0:
                break
        return self
    