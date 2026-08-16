import sys
sys.path.insert(0, '.')
import numpy as np
from utils.data_utils import train_test_split_per_class, standardize

class_dict = {
    1: np.random.randn(500, 2),
    2: np.random.randn(500, 2) + 5,
    3: np.random.randn(500, 2) - 5,
}

X_train, y_train, X_test, y_test = train_test_split_per_class(class_dict)

print("X_train shape:", X_train.shape)
print("X_test shape:", X_test.shape)
print("Train class counts:", np.unique(y_train, return_counts=True))
print("Test class counts:", np.unique(y_test, return_counts=True))