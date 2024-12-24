import numpy as np

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
    """
    n_samples, n_features = X.shape
    # (1) Initialize w to the zero vector
    w = np.zeros(n_features)

    # (2) Loop over rounds
    for round_t in range(1, max_rounds + 1):
        mistake_found = False

        # (3) Loop over each sample
        for i in range(n_samples):
            x_i = X[i]
            y_i = y[i]

            # Condition: w . x_i <= gamma/2 but label is +1 => update
            if np.dot(w, x_i) <= gamma / 2 and y_i == +1:
                w = w + x_i
                mistake_found = True
                break  # proceed to next round

            # Condition: w . x_i >= gamma/2 but label is -1 => update
            elif np.dot(w, x_i) >= gamma / 2 and y_i == -1:
                w = w - x_i
                mistake_found = True
                break  # proceed to next round

        # If no mistakes, we have convergence
        if not mistake_found:
            print(f"No mistakes in round {round_t}. Algorithm converged.")
            return w, round_t

    print(f"Reached max rounds ({max_rounds}) without perfect convergence.")
    return w, max_rounds


if __name__ == "__main__":
    # 1. Read the data from iris.txt
    #    Make sure iris.txt is in the same folder as this script.
    X_list = []
    y_list = []

    # Define the classes to use
    class_a = "versicolor"
    class_b = "virginica"

    with open("iris.txt", "r") as f:
        for line in f:
            line = line.strip()
            if not line:  # skip blank lines
                continue

            parts = line.split()
            # The first 4 columns are features
            sepal_length = float(parts[0])
            sepal_width  = float(parts[1])  # 2nd feature
            petal_length = float(parts[2])  # 3rd feature
            petal_width  = float(parts[3])
            label_str    = parts[4]  # e.g. "Iris-setosa"

            # Dynamically classify based on user-defined class_a and class_b
            if class_a in label_str:
                label = +1
            elif class_b in label_str:
                label = -1
            else:
                continue  # Skip other classes

            # Use only the 2nd and 3rd features
            X_list.append([sepal_width, petal_length])
            y_list.append(label)

    X = np.array(X_list, dtype=float)
    y = np.array(y_list, dtype=float)

    # 2. Run the perceptron with a chosen margin
    gamma = 0.5
    w_final, t_final = perceptron_with_margin(X, y, gamma, max_rounds=1000)

    print("Final weight vector:", w_final)
    print("Algorithm finished on round:", t_final)
