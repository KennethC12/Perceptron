import numpy as np
import matplotlib.pyplot as plt

# These are helper functions for plotting the decision boundary
def plot_bounds_from_wb(w, b, X, label=None): # This function plots the bounds of the decision boundary with weight vector w and bias b
    w1, w2 = w
    xs = np.linspace(X[:,0].min()-.5, X[:,0].max()+.5, 200) # This evenly spreads out the points in the x-axis

    if abs(w2) > 1e-12:
        ys = -(b+w1*xs) / (w2 + 1e-12) # Add a small epsilon to avoid division by zero
        plt.plot(xs, ys, label=label) # Plot the line
    elif abs(w1) > 1e-12:
        x_vert = -b/(w1 + 1e-12) # Calculate the x-coordinate of the vertical line
        plt.axvline(x_vert, label=label) # Plot the axis vertical line

def plot_boundary_aug(w_aug, X, label=None): # This function plots the boundary of the decision boundary with weight vector w_aug
    """Augmented boundary for w_aug=[b,w1,w2]."""
    b, w1, w2 = w_aug
    xs = np.linspace(X[:,0].min()-0.5, X[:,0].max()+0.5, 200) # This evenly spreads out the points in the x-axis

    if abs(w2) > 1e-12:
        ys = -(b + w1*xs)/(w2 + 1e-12)
        plt.plot(xs, ys, label=label)
    elif abs(w1) > 1e-12:
        x_vert = -b/(w1 + 1e-12)
        plt.axvline(x_vert, label=label)


# ---- Part A: Tiny World ----
"""
Discussion:
Created a small 
"""
X = np.array([[1,1], [2,2], [1,2], [-1,-1], [-1,-2], [-2,-2]], dtype=float)
y = np.array([1,1,1,-1,-1,-1], dtype=int)

w_chosen = np.array([1.0, 1.0])
b_chosen = 0.0

plt.figure(figsize=(5.5,5.5))
plt.scatter(X[y==1,0], X[y==1,1], marker='o', label='y=+1')
plt.scatter(X[y==-1,0], X[y==-1,1], marker='x', label='y=-1')
plot_bounds_from_wb(w_chosen, b_chosen, X, label='chosen (w,b)')
plt.xlabel('x1'); plt.ylabel('x2'); plt.grid(True); plt.legend(); plt.title('Part A: Tiny World')
plt.show()

# ---- Part B: Perceptron from Scratch ----
"""
Discussion:
Created Perceptron from scratch. It starts with a random weight and visits each point in the dataset one by one. 
At every step it checks if the point is on the wrong side of the hyperplane by using g(x) = b + w1*x1 + w2*x2.
If the point is wrong or the boundary of the hyperplane is not correct, it updates the weight and bias.

"""

class Perceptron:
    """
    Augmented weights: self.w = [b, w1, w2]
    We record a frame after EVERY sample visit (even if no update).
    history[k] contains:
      - 'w': weights *after* processing that sample (None for the initial frame),
      - 'i': index of the sample processed on this frame (-1 for initial),
      - 'seen': how many samples have been seen so far in this pass (for axis bounds),
      - 'updated': whether a mistake update happened.
    """
    
    def __init__(self, num_inputs = 2, lr = .1, seed = 0):
        self.lr = float(lr)
        rng = np.random.default_rng(seed) # The argument seed makes the randomness reproducible.
        self.w = rng.standard_normal(num_inputs + 1) # [b, w1, w2]
        self.history = []

    def score(self, x):
        return self.w[0] +self.w[1:].dot(x) # self.w[0] = b, self.w[1:] = [w1, w2] so self.w[0] + self.w[1:].dot(x) = b + w1*x1 + w2*x2

    def predict(self, x):
        return 1 if self.score(x) > 0 else -1 # This is the prediction rule for the perceptron

    def fit(self, X, y, epochs=1):
        self.history = []
        for _ in range(epochs):
            for i in range(len(X)):
                g = self.score(X[i]) # This is the score of the perceptron
                updated = False
                if y[i] * g <= 0:                 # mistake (or on boundary)
                    self.w[1:] += self.lr * y[i] * X[i]
                    self.w[0]  += self.lr * y[i]
                    updated = True
                    # record after visiting this sample
                self.history.append({"w": self.w.copy(), "i": i, "updated": updated})
            return self

p = Perceptron(num_inputs=2, lr=1.0, seed=0)
p.fit(X, y)
print("Final weights [b,w1,w2] after one pass:", np.round(p.w, 4))
print("Total mistakes (frames):", len(p.history))
print("Total updates:", sum(fr["updated"] for fr in p.history))

# Plotting the history
for k, fr in enumerate(p.history, start=1): 
    i = fr["i"]
    w_aug = fr["w"]

    plt.figure(figsize=(5.5,5.5))

    plt.scatter(X[:,0], X[:,1], alpha=0.5, label="points")

    X_seen, y_seen = X[:i+1], y[:i+1]

    plt.scatter(X_seen[y_seen == 1,0], X_seen[y_seen == 1, 1], marker='o', label='y=+1 seen') # Markers for positive samples
    plt.scatter(X_seen[y_seen == -1,0], X_seen[y_seen == -1, 1], marker='x', label='y=-1 seen') # Markers for negative samples

    plt.scatter([X[i,0]],[X[i,1]], s=140, facecolors="none", edgecolors="tab:red", linewidths=2, label="current") # Current sample
    if i+1 < len(X):
        j = i+1
        plt.scatter([X[j,0]],[X[j,1]], s=140, facecolors="none", edgecolors="gold", linewidths=2, label="next")
    # boundary + weight vector
    plot_boundary_aug(w_aug, X, label="boundary")
    b, w1, w2 = w_aug
    plt.arrow(0,0,w1,w2,head_width=0.15,length_includes_head=True)
    plt.text(w1,w2,"  w",va="center")

    # This adjusts the x and y axis to the minimum and maximum of the seen samples
    xmin, xmax = X_seen[:,0].min(), X_seen[:,0].max()
    ymin, ymax = X_seen[:,1].min(), X_seen[:,1].max()
    pad_x = 0.5 * max(1e-6, xmax - xmin)
    pad_y = 0.5 * max(1e-6, ymax - ymin)
    plt.xlim(xmin - pad_x, xmax + pad_x)
    plt.ylim(ymin - pad_y, ymax + pad_y)

    tag = " (updated)" if fr["updated"] else " (no update)"
    plt.title(f"Step {k}: after visiting sample {i}{tag}")
    plt.xlabel("x1"); plt.ylabel("x2"); plt.grid(True); plt.legend()
    plt.show()
    
# ---- Part C: Bound vs. Reality ----
"""
Discussion:
Bound vs Reality looks at how the perceptron;s performace comapres to its limits. 
The theorical formula says that k <= (R/𝛾)**2.
Based on what I computed on both caluses the real number of mistakes was smaller than the bound, 
which means that the bound is just a guarantee not an exact count.
This means that the perceptron is performing well and is not overfitting.
"""

# R = max_i || [x_i ; 1] || 2

X_aug = np.hstack([np.ones((X.shape[0], 1)), X]) 
R = np.linalg.norm(X_aug, axis=1).max()

wstar_aug = np.array([b_chosen, *w_chosen])
norm_wstar = np.linalg.norm(wstar_aug)
margins = y * (X_aug @ wstar_aug) / (norm_wstar + 1e-12)
gamma = margins.min()

k_mistakes = len(p.history)
bound = (R / (gamma + 1e-12))**2

print(f"R = {R:.4f}")
print(f"gamma (from chosen w*) = {gamma:.4f}")
print(f"(R/gamma)^2 = {bound:.4f}")
print(f"Actual mistakes k = {k_mistakes}")

# ---- Part D: When it Fails (XOR) + Feature Map Φ ----
"""
Discussion:
XOR is a dataset that is not linearly separable. 
It is beacuse the Perceptron never full conveges and it keeps udating the weights again and again.
This is why the Perceptron never converges on the XOR dataset.
So basically the Perceptron can only work on linearly separable datasets.
"""

X_xor = np.array([[0,0],[0,1],[1,0],[1,1]], dtype=float)
y_xor = np.array([-1, +1, +1, -1], dtype=int)

X_xor = np.array([[0,0],[0,1],[1,0],[1,1]], dtype=float)
y_xor = np.array([-1, +1, +1, -1], dtype=int)

# Show cycling (keeps making mistakes)
p_xor = Perceptron(num_inputs=2, lr=1.0, seed=1)
mistakes = 0
for epoch in range(10):  # a few epochs just to show it doesn't converge
    any_mistake = False
    for i in range(len(X_xor)):
        g = p_xor.score(X_xor[i])
        if y_xor[i]*g <= 0:
            p_xor.w[1:] += p_xor.lr*y_xor[i]*X_xor[i]
            p_xor.w[0]  += p_xor.lr*y_xor[i]
            mistakes += 1
            any_mistake = True
print("XOR mistakes across a few epochs (should be > 0):", mistakes)

# Polynomial feature map Φ(x) = [x1, x2, x1*x2, x1^2, x2^2]
def Phi(X):
    x1 = X[:,0]; x2 = X[:,1]
    return np.c_[x1, x2, x1*x2, x1**2, x2**2]

Xp = Phi(X_xor)
p_phi = Perceptron(num_inputs=Xp.shape[1], lr=1.0, seed=0)

def fit_until_converge(percep, Xf, y, max_epochs=100):
    for _ in range(max_epochs):
        errs = 0
        for i in range(len(Xf)):
            g = percep.w[0] + percep.w[1:].dot(Xf[i])
            if y[i]*g <= 0:
                percep.w[1:] += percep.lr*y[i]*Xf[i]
                percep.w[0]  += percep.lr*y[i]
                errs += 1
        if errs == 0:
            return True
    return False

converged = fit_until_converge(p_phi, Xp, y_xor, max_epochs=100)
print("Perceptron in Φ-space converged:", converged)

# Visualize implied nonlinear boundary back in original 2D: w^T Φ(x) + b = 0
xx, yy = np.meshgrid(np.linspace(-0.5,1.5,300), np.linspace(-0.5,1.5,300))
grid = np.c_[xx.ravel(), yy.ravel()]
Z = p_phi.w[0] + Phi(grid) @ p_phi.w[1:]
Z = Z.reshape(xx.shape)

plt.figure(figsize=(5.5,5.5))
plt.contour(xx, yy, Z, levels=[0], linewidths=2)  # decision contour
plt.scatter(X_xor[y_xor==1,0], X_xor[y_xor==1,1], marker='o', label='y=+1')
plt.scatter(X_xor[y_xor==-1,0], X_xor[y_xor==-1,1], marker='x', label='y=-1')
plt.xlim(-0.5, 1.5); plt.ylim(-0.5, 1.5)
plt.xlabel('x1'); plt.ylabel('x2'); plt.grid(True); plt.legend()
plt.title('Part D: Φ-space boundary mapped back to 2D')
plt.show()

# ---------- Part E: Scaling ----------
"""
Discussion:
In this part I scaled the input size of R, which showed how many updates are needed to converge.
Even though the shape of the data stays the same, larger values  can make learning slower because updates become larger in size.
This part shows that the perceptron’s speed and stability depend on how the input data is scaled.

"""

# Standardize features (z-score) column-wise
X_mean = X.mean(axis=0)
X_std  = X.std(axis=0) + 1e-12
X_stdzd = (X - X_mean) / X_std

# Recompute R, gamma on standardized data using a simple chosen direction
w_chosen_std = np.array([1.0, 1.0])
b_chosen_std = 0.0
X_aug_std = np.hstack([np.ones((X_stdzd.shape[0],1)), X_stdzd])
w_star_aug_std = np.array([b_chosen_std, *w_chosen_std])
R_std = np.linalg.norm(X_aug_std, axis=1).max()
gamma_std = (y * (X_aug_std @ w_star_aug_std) / (np.linalg.norm(w_star_aug_std)+1e-12)).min()

# Train perceptron on standardized data (one pass) to compare mistake count
p_std = Perceptron(num_inputs=2, lr=1.0, seed=0)
p_std.fit(X_stdzd, y)

print(f"Unscaled:     R={R:.4f},  gamma={gamma:.4f},  (R/gamma)^2={(R/(gamma+1e-12))**2:.4f},  mistakes={k_mistakes}")
print(f"Standardized: R={R_std:.4f}, gamma={gamma_std:.4f}, (R/gamma)^2={(R_std/(gamma_std+1e-12))**2:.4f}, mistakes={len(p_std.history)}")

# Final boundary in standardized space
plt.figure(figsize=(5.5,5.5))
plt.scatter(X_stdzd[y==1,0], X_stdzd[y==1,1], marker='o', label='y=+1 (std)')
plt.scatter(X_stdzd[y==-1,0], X_stdzd[y==-1,1], marker='x', label='y=-1 (std)')
plot_boundary_aug(p_std.w, X_stdzd, label='perceptron (std)')
plt.xlabel('x1 (std)'); plt.ylabel('x2 (std)'); plt.grid(True); plt.legend()
plt.title('Part E: Final boundary on standardized data')
plt.show()