#!/usr/bin/env python3
# Python 3.8 or higher

import numpy as np
from AdaBoost import (
    read_iris_data_two_classes,
    split_train_test,
    generate_all_lines,
    adaboost_train,
    compute_errors_by_iteration
)


def run_adaboost_experiment_100(filepath="iris.txt",
                                class_a="versicolor",
                                class_b="virginica",
                                test_size=0.5,
                                K=8,
                                NUM_RUNS=100):
    """
    Runs AdaBoost 100 times on the same two-class Iris subset (2 features),
    each time splitting train/test (50/50) randomly, then training for K rounds.
    Averages the train/test errors over the 100 runs and prints them out.
    """
    # Load entire dataset
    X, y = read_iris_data_two_classes(filepath=filepath,
                                      class_a=class_a,
                                      class_b=class_b)

    # We'll accumulate training & test errors for k=1..K
    train_errors_accum = np.zeros(K)
    test_errors_accum = np.zeros(K)

    for run_idx in range(NUM_RUNS):
        # 1. Split randomly (seed=None => truly random each time)
        X_train, y_train, X_test, y_test = split_train_test(X, y,
                                                            test_size=test_size,
                                                            seed=None)
        # 2. Generate all lines from training set
        lines = generate_all_lines(X_train)

        # 3. AdaBoost train
        selected_lines, alpha_list = adaboost_train(lines, X_train, y_train, K=K)

        # 4. Compute errors for partial ensembles
        train_errors, test_errors = compute_errors_by_iteration(
            selected_lines, alpha_list,
            X_train, y_train, X_test, y_test
        )

        # 5. Accumulate
        train_errors_accum += np.array(train_errors)
        test_errors_accum += np.array(test_errors)

    # Compute average
    train_errors_mean = train_errors_accum / NUM_RUNS
    test_errors_mean = test_errors_accum / NUM_RUNS

    # Print final results
    print(f"Results for {class_a} vs. {class_b} (over {NUM_RUNS} runs, K={K})")
    for k in range(1, K + 1):
        print(f"H{k}: avg train error={train_errors_mean[k-1]:.3f}, "
              f"avg test error={test_errors_mean[k-1]:.3f}")


if __name__ == "__main__":
    # Example usage
    run_adaboost_experiment_100(
        filepath="iris.txt",
        class_a="versicolor",
        class_b="virginica",
        test_size=0.5,
        K=8,
        NUM_RUNS=100
    )
