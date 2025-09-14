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
        self.history = []

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
        self.history = [None] # This creates a copy of the weights
        for _ in range(epochs):
            for xi, yi in zip(X, y):
                self.update_weights(xi, yi)
                self.history.append(self.weights.copy())
        return self
    
    def plot_step(self, X, y, step):
        plt.clf()
        pos = X[y == 1]; neg = X[y == -1]
        plt.scatter(pos[:, 0], pos[:, 1], marker="o", label="y=+1")
        plt.scatter(neg[:, 0], neg[:, 1], marker="x", label="y=-1")

        if step > 0:  # only draw boundary after first update
            w0, w1, w2 = self.history[step]
            x_min, x_max = X[:, 0].min() - 0.5, X[:, 0].max() + 0.5
            xs = np.linspace(x_min, x_max, 200)
            if abs(w2) > 1e-9:
                ys = -(w0 + w1 * xs) / w2
                plt.plot(xs, ys, label=f"boundary step {step}")
            elif abs(w1) > 1e-9:
                plt.axvline(-w0 / w1, label=f"boundary step {step}")

        plt.title(f"Perceptron Step {step}")
        plt.xlabel("x1"); plt.ylabel("x2")
        plt.legend(); plt.grid(True); plt.tight_layout()
        plt.draw()

# ---- Demo ----
X = np.array([[1,1],[2,2],[1,2],[-1,-1],[-1,-2],[-2,-2]])
y = np.array([1,1,1,-1,-1,-1])

p = Perceptron(num_input=2)
p.fit(X, y, epochs=2)  # record all updates

step = [0]
fig = plt.figure()

def on_key(event):
    if event.key == "right" and step[0] < len(p.history) - 1:
        step[0] += 1
        p.plot_step(X, y, step[0])
    elif event.key == "left" and step[0] > 0:
        step[0] -= 1
        p.plot_step(X, y, step[0])

fig.canvas.mpl_connect("key_press_event", on_key)
p.plot_step(X, y, step[0])
plt.show()