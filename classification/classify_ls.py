import matplotlib
matplotlib.use('Agg')   # taaki bina GUI ke bhi plot save ho jaye
import matplotlib.pyplot as plt
import os

os.makedirs('../plots', exist_ok=True)   # plots folder banao agar nahi hai
import sys
sys.path.insert(0, '..')
import itertools
import numpy as np
from utils.perceptron import Perceptron
from utils.data_utils import train_test_split_per_class, standardize
from utils.metrics import confusion_matrix, precision_recall_f1, accuracy

# ---------- Step 1: Data load ----------
class1 = np.loadtxt('../Group19/Group19/Classification/LS_Group19/Class1.txt')
class2 = np.loadtxt('../Group19/Group19/Classification/LS_Group19/Class2.txt')
class3 = np.loadtxt('../Group19/Group19/Classification/LS_Group19/Class3.txt')
class_dict = {1: class1, 2: class2, 3: class3}

# ---------- Step 2: Split (SIRF EK BAAR, seed=1 fixed) ----------
X_train, y_train, X_test, y_test = train_test_split_per_class(class_dict, train_frac=0.7, seed=1)
X_train, X_test = standardize(X_train, X_test)

classes = [1, 2, 3]


def encode_target(y_pair, class_a, activation):
    """class_a wale examples ko 'positive' class maanenge"""
    if activation == 'sigmoid':
        return np.where(y_pair == class_a, 1, 0)
    else:  # tanh
        return np.where(y_pair == class_a, 1, -1)


def train_all_pairs(X_train, y_train, activation):
    """Teeno pairs (1v2, 1v3, 2v3) train karta hai, dict me return karta hai"""
    models = {}
    for a, b in itertools.combinations(classes, 2):
        mask = np.isin(y_train, [a, b])
        X_pair = X_train[mask]
        y_pair = y_train[mask]
        target = encode_target(y_pair, a, activation)

        model = Perceptron(activation=activation, lr=0.05, epochs=500, seed=1)
        model.fit(X_pair, target)
        models[(a, b)] = model
        print(f"  Trained {a} vs {b} | final error: {model.error_history[-1]:.5f}")
    return models


def predict_one_vs_one(models, X):
    """Voting se final class predict karta hai"""
    votes = np.zeros((X.shape[0], len(classes)), dtype=int)
    class_idx = {c: i for i, c in enumerate(classes)}

    for (a, b), model in models.items():
        raw = model.predict_raw(X)
        if model.activation_name == 'sigmoid':
            pred_a = raw >= 0.5     # True matlab class 'a' jeeta
        else:
            pred_a = raw >= 0.0

        votes[pred_a, class_idx[a]] += 1
        votes[~pred_a, class_idx[b]] += 1

    winner_idx = np.argmax(votes, axis=1)
    return np.array([classes[i] for i in winner_idx])

def plot_error_vs_epoch(models, activation, dataset_name='LS'):
    plt.figure(figsize=(6, 4))
    for (a, b), model in models.items():
        plt.plot(model.error_history, label=f'{a} vs {b}')
    plt.xlabel('Epoch')
    plt.ylabel('Average Error')
    plt.title(f'{dataset_name} - {activation} - Error vs Epoch')
    plt.legend()
    plt.tight_layout()
    fname = f'../plots/{dataset_name.lower()}_error_vs_epoch_{activation}.png'
def plot_pairwise_decision_regions(models, X_train, y_train, activation, dataset_name='LS'):
    for (a, b), model in models.items():
        mask = np.isin(y_train, [a, b])
        Xp, yp = X_train[mask], y_train[mask]

        x_min, x_max = Xp[:, 0].min() - 1, Xp[:, 0].max() + 1
        y_min, y_max = Xp[:, 1].min() - 1, Xp[:, 1].max() + 1
        xx, yy = np.meshgrid(np.linspace(x_min, x_max, 300),
                              np.linspace(y_min, y_max, 300))
        grid = np.c_[xx.ravel(), yy.ravel()]

        raw = model.predict_raw(grid)
        if activation == 'sigmoid':
            pred = (raw >= 0.5).astype(int)
        else:
            pred = (raw >= 0.0).astype(int)
        pred = pred.reshape(xx.shape)

        plt.figure(figsize=(5, 4))
        plt.contourf(xx, yy, pred, alpha=0.3, cmap='coolwarm')
        plt.scatter(Xp[yp == a, 0], Xp[yp == a, 1], label=f'Class {a}', s=15, edgecolor='k')
        plt.scatter(Xp[yp == b, 0], Xp[yp == b, 1], label=f'Class {b}', s=15, edgecolor='k')
        plt.xlabel('x1'); plt.ylabel('x2')
        plt.title(f'{dataset_name} - {activation} - Decision region {a} vs {b}')
        plt.legend()
        plt.tight_layout()
        fname = f'../plots/{dataset_name.lower()}_decision_{activation}_{a}v{b}.png'
        plt.savefig(fname, dpi=150)
        plt.close()
        print("Saved:", fname)


def plot_combined_decision_region(models, X_train, y_train, activation, dataset_name='LS'):
    x_min, x_max = X_train[:, 0].min() - 1, X_train[:, 0].max() + 1
    y_min, y_max = X_train[:, 1].min() - 1, X_train[:, 1].max() + 1
    xx, yy = np.meshgrid(np.linspace(x_min, x_max, 300),
                          np.linspace(y_min, y_max, 300))
    grid = np.c_[xx.ravel(), yy.ravel()]

    pred = predict_one_vs_one(models, grid).reshape(xx.shape)

    plt.figure(figsize=(5.5, 4.5))
    plt.contourf(xx, yy, pred, alpha=0.3, cmap='viridis')
    colors = ['blue', 'red', 'green']
    for i, c in enumerate(classes):
        pts = X_train[y_train == c]
        plt.scatter(pts[:, 0], pts[:, 1], label=f'Class {c}', s=15,
                    edgecolor='k', c=colors[i])
    plt.xlabel('x1'); plt.ylabel('x2')
    plt.title(f'{dataset_name} - {activation} - Combined Decision Region')
    plt.legend()
    plt.tight_layout()
    fname = f'../plots/{dataset_name.lower()}_decision_combined_{activation}.png'
    plt.savefig(fname, dpi=150)
    plt.close()
    print("Saved:", fname)
# ---------- Step 3: Train + Evaluate for BOTH activations ----------
if __name__ == "__main__":
  for activation in ['sigmoid', 'tanh']:
    print(f"\n===== Activation: {activation} =====")
    models = train_all_pairs(X_train, y_train, activation)
    plot_error_vs_epoch(models, activation, dataset_name='LS')
    plot_pairwise_decision_regions(models, X_train, y_train, activation, dataset_name='LS')
    plot_combined_decision_region(models, X_train, y_train, activation, dataset_name='LS')

    y_pred = predict_one_vs_one(models, X_test)

    cm = confusion_matrix(y_test, y_pred, labels=classes)
    p, r, f1 = precision_recall_f1(cm)
    acc = accuracy(cm)

    print("Confusion Matrix:\n", cm)
    print("Accuracy:", acc)
    print("Precision per class:", p, "| Mean:", p.mean())
    print("Recall per class:", r, "| Mean:", r.mean())
    print("F1 per class:", f1, "| Mean:", f1.mean())