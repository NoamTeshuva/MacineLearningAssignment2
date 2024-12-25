import numpy as np
from itertools import combinations

def perceptron_with_margin(X, y, gamma, max_rounds=1000):
    """
    Perceptron algorithm with a margin-based update rule.

    Parameters:
    -----------
    X : np.ndarray
        Data matrix of shape (n_samples, n_features).
    y : np.ndarray
        Labels of shape (n_samples,), each label is +1 or -1.
    gamma : float
        The margin parameter for the update checks, e.g. w_t . x_i <= gamma/2, etc.
    max_rounds : int, optional
        Maximum number of rounds to run before giving up (to prevent infinite loops).

    Returns:
    --------
    w : np.ndarray
        Final weight vector after training.
    round_t : int
        The last round index when the algorithm finished (for informational purposes).
    mistakes : int
        Total number of mistakes made by the algorithm.
    """
    n_samples, n_features = X.shape
    w = np.zeros(n_features)  # Initialize w to the zero vector
    mistakes = 0

    for round_t in range(1, max_rounds + 1):
        mistake_found = False

        for i in range(n_samples):
            x_i = X[i]
            y_i = y[i]

            # Update conditions
            if np.dot(w, x_i) <= gamma / 2 and y_i == +1:
                w = w + x_i
                mistakes += 1
                mistake_found = True
                break
            elif np.dot(w, x_i) >= gamma / 2 and y_i == -1:
                w = w - x_i
                mistakes += 1
                mistake_found = True
                break

        if not mistake_found:
            print(f"No mistakes in round {round_t}. Algorithm converged.")
            return w, round_t, mistakes

    print(f"Reached max rounds ({max_rounds}) without perfect convergence.")
    return w, max_rounds, mistakes

def brute_force_max_margin(X, y):
    """
    Brute force algorithm to find the maximum margin hyperplane using all possible combinations of points.

    Parameters:
    -----------
    X: np.ndarray
        Feature matrix (n_samples x n_features).
    y: np.ndarray
        Labels (+1, -1) for the dataset.

    Returns:
    --------
    best_w: np.ndarray
        Normal vector of the hyperplane with the largest margin.
    best_b: float
        Bias term for the hyperplane with the largest margin.
    max_margin: float
        The largest margin found.
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

if __name__ == "__main__":
    # 1. Read the data from iris.txt
    X_list = []
    y_list = []

    # Define the classes to use
    class_a = "versicolor"
    class_b = "setosa"

    with open("iris.txt", "r") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue

            parts = line.split()
            sepal_width = float(parts[1])  # 2nd feature
            petal_length = float(parts[2])  # 3rd feature
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

    # 2. Run the perceptron with a chosen margin
    gamma = 0.5
    w_final, t_final, total_mistakes = perceptron_with_margin(X, y, gamma, max_rounds=1000)

    print("Final weight vector from Perceptron:", w_final)
    print("Algorithm finished on round:", t_final)
    print("Total mistakes made:", total_mistakes)

    # 3. Find the hyperplane with the largest margin using brute force
    best_w, best_b, max_margin = brute_force_max_margin(X, y)

    print("Best hyperplane normal vector (w):", best_w)
    print("Best hyperplane bias (b):", best_b)
    print("Maximum margin:", max_margin)
