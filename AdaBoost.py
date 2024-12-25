import numpy as np
from itertools import combinations
import random

###################################
# 1. Read and filter the Iris data
###################################
def read_iris_data_two_classes(filepath="iris.txt", class_a="versicolor", class_b="setosa"):
    """
    Reads iris data from a file, returning X (two features) and y (in {+1, -1})
    for the specified two classes.
    """
    X_list = []
    y_list = []

    with open(filepath, "r") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue

            parts = line.split()
            # Features at indices 1 and 2 (sepal_width, petal_length)
            sepal_width = float(parts[1])   # 2nd feature
            petal_length = float(parts[2]) # 3rd feature
            label_str = parts[4]

            # Filter for exactly two classes
            if class_a in label_str:
                label = +1
            elif class_b in label_str:
                label = -1
            else:
                # Skip anything else (e.g., virginica)
                continue

            X_list.append([sepal_width, petal_length])
            y_list.append(label)

    X = np.array(X_list, dtype=float)
    y = np.array(y_list, dtype=float)
    return X, y

###############################
# 2. Split the data train/test
###############################
def split_train_test(X, y, test_size=0.5, seed=42):
    """
    Splits X, y into train and test sets.
    test_size fraction goes to test, the rest to train.
    """
    np.random.seed(seed)
    indices = np.arange(len(X))
    np.random.shuffle(indices)

    split_idx = int(len(X) * (1 - test_size))
    train_indices = indices[:split_idx]
    test_indices = indices[split_idx:]

    X_train, y_train = X[train_indices], y[train_indices]
    X_test, y_test = X[test_indices], y[test_indices]
    return X_train, y_train, X_test, y_test

###################################################
# 3. Generate hypothesis set: all lines through
#    pairs of training points in 2D
###################################################
def generate_all_lines(X_train):
    """
    Generate all lines (weak classifiers) from pairs of points in X_train.
    Each line is represented by (w, b), where pred(x) = sign(w^T x + b).
    We return a list of such (w, b) pairs.
    """
    lines = []
    n = len(X_train)

    # All pairs i,j
    for i, j in combinations(range(n), 2):
        p = X_train[i]  # [x1, x2]
        q = X_train[j]  # [x1, x2]
        if np.allclose(p, q):
            continue

        # Normal to the line through p and q in 2D:
        # direction = q - p => normal = (dy, -dx)
        dx = q[0] - p[0]
        dy = q[1] - p[1]

        # normal vector w
        w = np.array([dy, -dx], dtype=float)

        # Solve for b: line passes through p => w^T p + b = 0 => b = - w^T p
        b = -np.dot(w, p)

        # We store both orientations: (w, b) and (-w, -b)
        lines.append((w, b))
        lines.append((-w, -b))

    return lines

#####################################
# 4. Helper: predict with a line (w,b)
#####################################
def predict_line(line, X):
    """
    line: (w, b)
    X: array of shape (n_samples, 2)
    Returns array of predictions in {+1, -1} for each row in X.
    """
    (w, b) = line
    # raw_scores = w^T x + b
    raw_scores = X @ w + b  # shape (n_samples,)
    preds = np.sign(raw_scores)
    # By definition, sign(0) can be 0, so let's force it to +1 if it's 0
    preds[preds == 0] = 1
    return preds

######################################################
# 5. Find best weak learner among all lines for AdaBoost
######################################################
def find_best_weak_learner(lines, X_train, y_train, D):
    """
    lines: list of (w, b) candidates
    X_train: shape (n, 2)
    y_train: shape (n,)
    D: shape (n,) = distribution over training examples

    Returns:
        best_line: The line with the minimal weighted error
        min_error: Weighted classification error of best_line
    """
    n = len(X_train)
    best_line = None
    min_error = float('inf')

    for line in lines:
        preds = predict_line(line, X_train)
        misclassified = (preds != y_train)
        error = np.sum(D[misclassified])
        if error < min_error:
            min_error = error
            best_line = line

    return best_line, min_error

################################
# 6. The AdaBoost Training Loop
################################
def adaboost_train(lines, X_train, y_train, K=8):
    n = len(X_train)
    D = np.ones(n) / n

    selected_lines = []
    alpha_list = []

    for t in range(K):
        h_t, epsilon_t = find_best_weak_learner(lines, X_train, y_train, D)

        # 1) Clip epsilon_t away from 0 or 1
        eps_clip = max(min(epsilon_t, 1 - 1e-12), 1e-12)
        alpha_t = 0.5 * np.log((1 - eps_clip) / eps_clip)

        selected_lines.append(h_t)
        alpha_list.append(alpha_t)

        preds = predict_line(h_t, X_train)
        update_factor = np.exp(-alpha_t * (y_train * preds))
        D *= update_factor

        # 2) Safe normalize
        Z = np.sum(D)
        if Z < 1e-15:
            # fallback if sum(D) is effectively 0
            D = np.ones(n) / n
        else:
            D /= Z

    return selected_lines, alpha_list


###################################################
# 7. Predict using the ensemble up to iteration k
###################################################
def adaboost_predict_ensemble(X, lines, alphas):
    """
    Predicts using the ensemble of weak classifiers 'lines' with weights 'alphas'.

    lines: list of (w, b)
    alphas: list of float, same length as lines
    X: shape (n_samples, 2)

    Returns:
        preds: shape (n_samples,) in {+1, -1}
    """
    scores = np.zeros(len(X))
    for line, alpha in zip(lines, alphas):
        h_preds = predict_line(line, X)
        scores += alpha * h_preds
    final_preds = np.sign(scores)
    final_preds[final_preds == 0] = 1
    return final_preds

#################################################################
# 8. Compute training/test errors for partial ensembles H_k
#################################################################
def compute_errors_by_iteration(selected_lines, alpha_list, X_train, y_train, X_test, y_test):
    """
    For k=1..K, build H_k from first k lines/alphas and compute training/test errors.
    Returns two lists: training_errors, test_errors for each k.
    """
    train_errors = []
    test_errors = []

    for k in range(1, len(selected_lines) + 1):
        # partial ensemble up to index k
        lines_k = selected_lines[:k]
        alphas_k = alpha_list[:k]

        # training error
        train_preds_k = adaboost_predict_ensemble(X_train, lines_k, alphas_k)
        train_err_k = np.mean(train_preds_k != y_train)
        train_errors.append(train_err_k)

        # test error
        test_preds_k = adaboost_predict_ensemble(X_test, lines_k, alphas_k)
        test_err_k = np.mean(test_preds_k != y_test)
        test_errors.append(test_err_k)

    return train_errors, test_errors

##########################################
# Main: Tie it all together & run AdaBoost
##########################################
if __name__ == "__main__":
    # 1. Read the data from iris.txt
    X, y = read_iris_data_two_classes(
        filepath="iris.txt",
        class_a="versicolor",  # +1
        class_b="virginica"       # -1
    )

    # 2. Split the data (50% train, 50% test)
    X_train, y_train, X_test, y_test = split_train_test(X, y, test_size=0.5, seed=42)

    # 3. Generate all lines (hypothesis set) from training points
    lines = generate_all_lines(X_train)
    print(f"Number of lines (weak classifiers) in hypothesis set: {len(lines)}")

    # 4. Train AdaBoost for K=8 rounds
    K = 8
    selected_lines, alpha_list = adaboost_train(lines, X_train, y_train, K=K)

    # 5. Compute training & test errors for H_k, k=1..K
    train_errors, test_errors = compute_errors_by_iteration(
        selected_lines, alpha_list, X_train, y_train, X_test, y_test
    )

    # 6. Print results
    for k in range(1, K + 1):
        print(f"k={k}: Train Error={train_errors[k-1]:.3f}, Test Error={test_errors[k-1]:.3f}")
