# Neural Networks — Assignment I: The Perceptron

**Instructor:** Tales Imbiriba  
**Due:** Sep. 17 (Wednesday) by 11:59 PM

---

## 🎯 Goal
Implement the **Perceptron algorithm** from scratch using only NumPy and Matplotlib, and explore:
- How the algorithm learns in real-time
- Experiments with linear separability and feature mappings
- The perceptron convergence theorem

---

## ✅ Rules
- Allowed libraries: **NumPy, Matplotlib**
- Do **not** use scikit-learn for the perceptron itself
- Labels must be in **{−1, +1}**
- Use the **augmented representation**:  
  \(\tilde{x} = [x; 1], \; \tilde{w} = [w; b]\)

---

## 📝 Grading Breakdown
| Part | Description             | Weight |
|------|-------------------------|--------|
| A    | Tiny World              | 15%    |
| B    | Perceptron from Scratch | 40%    |
| C    | Bound vs. Reality       | 20%    |
| D    | When it Fails           | 15%    |
| E    | Play with Scaling       | 10%    |

---

## 📂 Assignment Parts

### Part A — Tiny World (15%)
1. Create your own **6-point, 2D, linearly separable dataset** with labels in {−1, +1}.  
2. Plot the two classes and state one separating hyperplane (w, b).  

**Deliverable:** scatter plot + chosen (w, b).

---

### Part B — Perceptron from Scratch (40%)
- Implement **augmented perceptron** with mistake-driven updates:  
  \(\tilde{w} \leftarrow \tilde{w} + \eta y \tilde{x}\).  
- After every mistake, draw:
  - all data points (highlight current + next ones)
  - decision boundary \(w^\top x + b = 0\)
  - weight vector \(w\) as an arrow from origin  

**Deliverable:** sequence of plots + final (w, b).

---

### Part C — Bound vs. Reality (20%)
1. Compute \(R = \max_i \|[x_i;1]\|_2\).  
2. Estimate margin \(\gamma\):  
   \(\gamma = \min_i \dfrac{y_i \tilde{w}^{*\top} \tilde{x}_i}{\|\tilde{w}^*\|_2}\).  
3. Compare actual mistakes \(k\) to theorem’s bound \((R/\gamma)^2\).  

**Deliverable:** table with R, γ, (R/γ)², k + discussion.

---

### Part D — When it Fails (15%)
1. Build a **non-separable 2D set** (e.g., XOR).  
2. Run perceptron → show it does **not converge** (cycles).  
3. Add a hand-crafted feature map Φ (e.g., polynomial features) and rerun perceptron.  
4. Visualize implied boundary back in 2D as contour of \(w^\top \Phi(x) + b = 0\).  

**Deliverable:** plots of failure (original space) + success (feature space), and a short explanation of why Φ helps.

---

### Part E — Play with Scaling (10%)
- Standardize or rescale features.  
- Recompute R, estimate γ, and rerun perceptron.  
- Discuss how scaling affected R, γ, and convergence speed.  

---

## 🛠️ Starter Snippets

**Compute R (with augmentation):**
```python
import numpy as np
X_aug = np.hstack([X, np.ones((X.shape[0], 1))])
R = np.linalg.norm(X_aug, axis=1).max()
