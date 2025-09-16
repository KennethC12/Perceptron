import numpy as np
import matplotlib.pyplot as plt

# Part A: Tiny World
def plot_boundary(w, X, label=None):
    """Draw decision boundary for augmented weights w=[b,w1,w2]."""
    b, w1, w2 = w
    xs = np.linspace(X[:,0].min()-0.5, X[:,0].max()+0.5, 200)
    if abs(w2) > 1e-12:
        ys = -(b + w1*xs) / (w2 + 1e-12)
        plt.plot(xs, ys, label=label)
    elif abs(w1) > 1e-12:
        x_vert = -b / (w1 + 1e-12)
        plt.axvline(x_vert, label=label)

# Part B: Perceptron from Scratch
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
    def __init__(self, num_input, learning_rate=1.0, seed=None):
        self.lr = float(learning_rate)
        if seed is not None:
            rng = np.random.default_rng(seed)
            self.w = rng.standard_normal(num_input + 1)
        else:
            self.w = np.random.randn(num_input + 1)
        self.history = []
        self.visit_order = []  # sequence of sample indices we visit

    def score(self, X):
        X = np.asarray(X)
        if X.ndim == 1:
            return self.w[0] + X @ self.w[1:]
        return self.w[0] + X @ self.w[1:]

    def predict(self, X):
        g = self.score(X)
        if np.ndim(g) == 0:
            return 1 if g >= 0 else -1
        return np.where(g >= 0, 1, -1)

    def _update_if_mistake(self, x, y):
        g = self.score(x)
        if y * g <= 0:  # count boundary as mistake
            self.w[1:] += self.lr * y * x
            self.w[0]   += self.lr * y
            return True
        return False

    def fit(self, X, y, epochs=1):
        n = len(X)
        self.history = []
        self.visit_order = []

        # Frame 0: before anything happens -> no line
        self.history.append({"w": None, "i": -1, "seen": 0, "updated": False})

        for _ in range(epochs):
            for i in range(n):
                self.visit_order.append(i)
                updated = self._update_if_mistake(X[i], y[i])
                # record a frame after visiting this sample (line shows current w)
                seen = (self.history[-1]["seen"] + 1)
                self.history.append({"w": self.w.copy(), "i": i, "seen": seen, "updated": updated})
        return self

    def plot_step(self, X, y, k):
        """
        k indexes frames in self.history.
        Frame 0: no boundary (w is None)
        Frame >=1: boundary for weights AFTER processing that frame's sample
        """
        plt.clf()

        # figure out what we've "seen" so far (for axis bounds)
        frame = self.history[k]
        seen = frame["seen"]  # number of points visited so far in current pass (0 at start)

        # scatter all points lightly in background
        plt.scatter(X[:,0], X[:,1], alpha=0.15, label="all points")

        # scatter just the seen points more prominently
        if seen > 0:
            X_seen = X[:seen]
            y_seen = y[:seen]
            pos = X_seen[y_seen == 1]
            neg = X_seen[y_seen == -1]
            if len(pos): plt.scatter(pos[:,0], pos[:,1], marker="o", label="y=+1 (seen)")
            if len(neg): plt.scatter(neg[:,0], neg[:,1], marker="x", label="y=-1 (seen)")
        else:
            X_seen = np.empty((0,2))

        # highlight: current (processed) and next
        i = frame["i"]
        if i >= 0:
            plt.scatter([X[i,0]], [X[i,1]], s=140, facecolors="none",
                        edgecolors="tab:red", linewidths=2, label="current")
            # next sample index (one-by-one within current epoch order)
            # Find where we are in visit_order to know the next
            # (k==0 has i=-1; from k>=1, the (k-1)-th visit is at visit_order[k-1])
            if k-1 < len(self.visit_order):
                pos_in_order = k-1
                if pos_in_order + 1 < len(self.visit_order):
                    nxt_idx = self.visit_order[pos_in_order + 1]
                    plt.scatter([X[nxt_idx,0]], [X[nxt_idx,1]], s=140, facecolors="none",
                                edgecolors="gold", linewidths=2, label="next")

        # draw boundary if we have weights for this frame
        if frame["w"] is not None:
            plot_boundary(frame["w"], X_seen if seen>0 else X, label="boundary")

            # draw weight vector arrow (from origin)
            b, w1, w2 = frame["w"]
            plt.arrow(0, 0, w1, w2, head_width=0.15, length_includes_head=True)
            plt.text(w1, w2, "  w", va="center")

        # dynamic bounds: fit to points seen so far (with padding).
        if seen > 0:
            xmin, xmax = X_seen[:,0].min(), X_seen[:,0].max()
            ymin, ymax = X_seen[:,1].min(), X_seen[:,1].max()
        else:
            xmin, xmax = X[:,0].min(), X[:,0].max()
            ymin, ymax = X[:,1].min(), X[:,1].max()

        pad_x = 0.5 * max(1e-6, xmax - xmin)
        pad_y = 0.5 * max(1e-6, ymax - ymin)
        plt.xlim(xmin - pad_x, xmax + pad_x)
        plt.ylim(ymin - pad_y, ymax + pad_y)

        upd_txt = " (updated)" if frame["updated"] else ""
        plt.title(f"Step {k}/{len(self.history)-1}  |  seen: {seen}{upd_txt}")
        plt.xlabel("x1"); plt.ylabel("x2")
        plt.grid(True); plt.legend(loc="best"); plt.tight_layout()
        plt.draw()


X = np.array([[1,1],[2,2],[1,2],[-1,-1],[-1,-2],[-2,-2]], dtype=float)
y = np.array([ 1,  1,  1, -1, -1, -1], dtype=int)

p = Perceptron(num_input=2, learning_rate=1.0, seed=0)
p.fit(X, y, epochs=1)   # visit each point once (one-by-one)

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
