import sys
sys.path.insert(0, '..')
import numpy as np

# classify_ls.py se saare reusable functions le lo
from classify_ls import (train_all_pairs, predict_one_vs_one,
                          plot_error_vs_epoch, plot_pairwise_decision_regions,
                          plot_combined_decision_region, classes)
from utils.data_utils import train_test_split_per_class, standardize
from utils.metrics import confusion_matrix, precision_recall_f1, accuracy

# ---------- Step 1: NLS data load (alag tarika) ----------
data = np.loadtxt('../Group19/Group19/Classification/NLS_Group19.txt', skiprows=1)

class1 = data[0:300]
class2 = data[300:800]
class3 = data[800:1800]
class_dict = {1: class1, 2: class2, 3: class3}

print("Class sizes:", {k: v.shape for k, v in class_dict.items()})

# ---------- Step 2: Split (same seed=1) ----------
X_train, y_train, X_test, y_test = train_test_split_per_class(class_dict, train_frac=0.7, seed=1)
X_train, X_test = standardize(X_train, X_test)

print("Train shape:", X_train.shape, "Test shape:", X_test.shape)

# ---------- Step 3: Train + Evaluate for BOTH activations ----------
for activation in ['sigmoid', 'tanh']:
    print(f"\n===== Activation: {activation} =====")
    models = train_all_pairs(X_train, y_train, activation)

    plot_error_vs_epoch(models, activation, dataset_name='NLS')
    plot_pairwise_decision_regions(models, X_train, y_train, activation, dataset_name='NLS')
    plot_combined_decision_region(models, X_train, y_train, activation, dataset_name='NLS')

    y_pred = predict_one_vs_one(models, X_test)

    cm = confusion_matrix(y_test, y_pred, labels=classes)
    p, r, f1 = precision_recall_f1(cm)
    acc = accuracy(cm)

    print("Confusion Matrix:\n", cm)
    print("Accuracy:", acc)
    print("Precision per class:", p, "| Mean:", p.mean())
    print("Recall per class:", r, "| Mean:", r.mean())
    print("F1 per class:", f1, "| Mean:", f1.mean())