#!/usr/bin/env python3
# Python 3.8 or higher

import numpy as np
import matplotlib.pyplot as plt
from AdaBoost import (
    read_iris_data_two_classes,
    split_train_test,
    generate_all_lines,
    adaboost_train,
    adaboost_predict_ensemble
)


def plot_adaboost_ensemble_boundary(X, y, lines, alphas, title="AdaBoost Decision Boundary"):
    """
    Plots the 2D data and the final AdaBoost ensemble decision boundary.

    The boundary is where ensemble_score(x) = 0, i.e. sum_{i=1..K} alpha_i * sign(...)=0.
    We'll draw a contour plot of the region predicted +1 vs -1.
    """
    plt.figure(figsize=(8, 6))

    # Plot the data points
    for class_value in np.unique(y):
        plt.scatter(X[y == class_value, 0], X[y == class_value, 1],
                    label=f"Class {int(class_value)}", alpha=0.8)

    # Create a mesh grid to evaluate ensemble
    x_min, x_max = X[:, 0].min() - 0.5, X[:, 0].max() + 0.5
    y_min, y_max = X[:, 1].min() - 0.5, X[:, 1].max() + 0.5
    xx, yy = np.meshgrid(np.linspace(x_min, x_max, 200),
                         np.linspace(y_min, y_max, 200))
    grid_points = np.c_[xx.ravel(), yy.ravel()]  # shape (40000, 2)

    # Compute ensemble scores on this grid
    # sum_i alpha_i * sign(w_i^T x + b_i)
    scores = np.zeros(len(grid_points))
    for (w, b), alpha in zip(lines, alphas):
        # line predictions on the grid
        raw = grid_points @ w + b
        line_preds = np.sign(raw)
        line_preds[line_preds == 0] = 1
        scores += alpha * line_preds

    # Now scores>0 => predicted +1, scores<0 => predicted -1
    Z = scores.reshape(xx.shape)

    # Plot the decision boundary: contour where Z=0
    plt.contourf(xx, yy, Z, levels=[-1e10, 0, 1e10], alpha=0.3, colors=('red', 'blue'))
    plt.contour(xx, yy, Z, levels=[0], colors='k', linewidths=1)

    plt.xlabel("Feature 1 (e.g. sepal_width)")
    plt.ylabel("Feature 2 (e.g. petal_length)")
    plt.title(title)
    plt.legend()
    plt.grid(True)
    plt.show()


if __name__ == "__main__":
    # 1. Read the data (e.g., versicolor vs virginica) from iris.txt
    class_a = "versicolor"  # +1
    class_b = "virginica"   # -1
    X, y = read_iris_data_two_classes(filepath="iris.txt", class_a=class_a, class_b=class_b)

    # 2. Split into 50% train, 50% test
    X_train, y_train, X_test, y_test = split_train_test(X, y, test_size=0.5, seed=42)

    # 3. Generate all lines (hypotheses) from training data
    lines = generate_all_lines(X_train)
    print(f"Number of lines in hypothesis set: {len(lines)}")

    # 4. Train AdaBoost for K=3
    K = 9
    selected_lines, alpha_list = adaboost_train(lines, X_train, y_train, K=K)

    # 5. Plot the decision boundary of the final ensemble (using all K)
    plot_adaboost_ensemble_boundary(X_train, y_train, selected_lines, alpha_list,
                                    title=f"AdaBoost with K={K} on Iris ({class_a} vs. {class_b})")
