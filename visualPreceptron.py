import numpy as np
import matplotlib.pyplot as plt
from itertools import combinations

def perceptron_with_margin(X, y, gamma, max_rounds=1000):
    """
    Perceptron algorithm with a margin-based update rule.
    """
    n_samples, n_features = X.shape
    w = np.zeros(n_features)  # Initialize w to the zero vector

    for round_t in range(1, max_rounds + 1):
        mistake_found = False

        for i in range(n_samples):
            x_i = X[i]
            y_i = y[i]

            # Update conditions
            if np.dot(w, x_i) <= gamma / 2 and y_i == +1:
                w = w + x_i
                mistake_found = True
                break
            elif np.dot(w, x_i) >= gamma / 2 and y_i == -1:
                w = w - x_i
                mistake_found = True
                break

        if not mistake_found:
            print(f"No mistakes in round {round_t}. Algorithm converged.")
            return w, round_t

    print(f"Reached max rounds ({max_rounds}) without perfect convergence.")
    return w, max_rounds

def brute_force_max_margin(X, y):
    """
    Brute force algorithm to find the maximum margin hyperplane using all possible combinations of points.
    """
    max_margin = -np.inf
    best_w, best_b = None, None

    def calculate_margin(w, b, X, y):
        distances = y * (np.dot(X, w) + b) / np.linalg.norm(w)
        return np.min(distances)

    for combo in combinations(range(len(X)), 2):
        try:
            p1, p2 = X[combo[0]], X[combo[1]]
            if y[combo[0]] != y[combo[1]]:
                orthogonal_vector = p2 - p1
                orthogonal_vector /= np.linalg.norm(orthogonal_vector)

                midpoint = (p1 + p2) / 2
                b = -np.dot(orthogonal_vector, midpoint)

                margin = calculate_margin(orthogonal_vector, b, X, y)

                if margin > max_margin:
                    max_margin = margin
                    best_w = orthogonal_vector
                    best_b = b
        except Exception as e:
            continue

    return best_w, best_b, max_margin

def plot_hyperplane(X, y, w, b, margin):
    """
    Visualize the dataset, the hyperplane, and the margins.
    """
    plt.figure(figsize=(8, 6))

    for class_value in np.unique(y):
        plt.scatter(
            X[y == class_value, 0],
            X[y == class_value, 1],
            label=f"Class {class_value}",
            alpha=0.8
        )

    x_min, x_max = X[:, 0].min() - 1, X[:, 0].max() + 1
    x_vals = np.linspace(x_min, x_max, 200)
    y_vals = -(w[0] * x_vals + b) / w[1]

    plt.plot(x_vals, y_vals, 'k-', label="Hyperplane")

    margin_plus = -(w[0] * x_vals + b + margin) / w[1]
    margin_minus = -(w[0] * x_vals + b - margin) / w[1]
    plt.plot(x_vals, margin_plus, 'k--', label="Margin +")
    plt.plot(x_vals, margin_minus, 'k--', label="Margin -")

    plt.xlabel("Feature 1")
    plt.ylabel("Feature 2")
    plt.legend()
    plt.grid(True)
    plt.title(f"Max Margin: {margin:.3f}")
    plt.show()

if __name__ == "__main__":
    X_list = []
    y_list = []

    class_a = "versicolor"
    class_b = "virginica"

    with open("iris.txt", "r") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue

            parts = line.split()
            sepal_width = float(parts[1])
            petal_length = float(parts[2])
            label_str = parts[4]

            if class_a in label_str:
                label = +1
            elif class_b in label_str:
                label = -1
            else:
                continue

            X_list.append([sepal_width, petal_length])
            y_list.append(label)

    X = np.array(X_list, dtype=float)
    y = np.array(y_list, dtype=float)

    gamma = 0.5
    w_final, t_final = perceptron_with_margin(X, y, gamma, max_rounds=1000)
    print("Final weight vector from Perceptron:", w_final)
    print("Algorithm finished on round:", t_final)

    best_w, best_b, max_margin = brute_force_max_margin(X, y)
    print("Best hyperplane normal vector (w):", best_w)
    print("Best hyperplane bias (b):", best_b)
    print("Maximum margin:", max_margin)

    if best_w is not None and best_b is not None:
        plot_hyperplane(X, y, best_w, best_b, max_margin)
    else:
        print("No valid hyperplane found for visualization.")
