import numpy as np

def train_test_split_per_class(class_dict, train_frac=0.7, seed=1):
    """
    class_dict: {label: array_of_points}  jaise {1: class1_data, 2: class2_data, 3: class3_data}
    Har class ko ALAG se split karta hai train/test me.
    """
    rng = np.random.RandomState(seed)
    X_train_list, y_train_list = [], []
    X_test_list, y_test_list = [], []

    for label, X in class_dict.items():
        n = X.shape[0]
        idx = rng.permutation(n)          # random order of indices
        n_train = int(round(n * train_frac))

        train_idx = idx[:n_train]
        test_idx = idx[n_train:]

        X_train_list.append(X[train_idx])
        y_train_list.append(np.full(len(train_idx), label))

        X_test_list.append(X[test_idx])
        y_test_list.append(np.full(len(test_idx), label))

    X_train = np.vstack(X_train_list)
    y_train = np.concatenate(y_train_list)
    X_test = np.vstack(X_test_list)
    y_test = np.concatenate(y_test_list)

    return X_train, y_train, X_test, y_test


def standardize(X_train, X_test):
    """
    Data ko normalize karta hai (mean=0, std=1) — gradient descent
    fast aur stable converge karta hai isse. Sirf TRAIN data ke stats use karo.
    """
    mu = X_train.mean(axis=0)
    sigma = X_train.std(axis=0)
    sigma[sigma == 0] = 1.0   # divide by zero se bachne ke liye

    X_train_scaled = (X_train - mu) / sigma
    X_test_scaled = (X_test - mu) / sigma

    return X_train_scaled, X_test_scaled