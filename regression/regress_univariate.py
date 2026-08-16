import sys
sys.path.insert(0, '..')
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import os

from utils.perceptron import Perceptron

os.makedirs('../plots', exist_ok=True)

# ---------- Step 1: Data load ----------
data = np.loadtxt('../Group19/Group19/Regression/UnivariateData/19.csv', delimiter=',')
X = data[:, [0]]   # shape (N,1)
y = data[:, 1]

print("X shape:", X.shape, "y shape:", y.shape)

# ---------- Step 2: Train/test split (simple random, seed=1 fixed) ----------
np.random.seed(1)
n = X.shape[0]
idx = np.random.permutation(n)
n_train = int(0.7 * n)
train_idx, test_idx = idx[:n_train], idx[n_train:]

X_train, y_train = X[train_idx], y[train_idx]
X_test, y_test = X[test_idx], y[test_idx]

print("Train:", X_train.shape, "Test:", X_test.shape)

# ---------- Step 3: Standardize X ----------
mu, sigma = X_train.mean(axis=0), X_train.std(axis=0)
X_train_s = (X_train - mu) / sigma
X_test_s = (X_test - mu) / sigma

# ---------- Step 4: Train linear perceptron ----------
model = Perceptron(activation='linear', lr=0.05, epochs=1000, seed=1)
model.fit(X_train_s, y_train)

y_pred_train = model.predict_raw(X_train_s)
y_pred_test = model.predict_raw(X_test_s)

print("Final training error:", model.error_history[-1])

# ---------- Step 5: RMSE and %RMSE ----------
def rmse(y_true, y_pred):
    return np.sqrt(np.mean((y_true - y_pred) ** 2))

def percent_rmse(y_true, y_pred):
    return 100 * rmse(y_true, y_pred) / np.mean(np.abs(y_true))

train_rmse = rmse(y_train, y_pred_train)
test_rmse = rmse(y_test, y_pred_test)
train_prmse = percent_rmse(y_train, y_pred_train)
test_prmse = percent_rmse(y_test, y_pred_test)

print(f"Train RMSE: {train_rmse:.4f}  | %RMSE: {train_prmse:.2f}%")
print(f"Test RMSE:  {test_rmse:.4f}  | %RMSE: {test_prmse:.2f}%")

# ---------- Step 6: Plot 1 - Error vs Epoch ----------
plt.figure(figsize=(6, 4))
plt.plot(model.error_history)
plt.xlabel('Epoch')
plt.ylabel('Average Error')
plt.title('Univariate Regression - Error vs Epoch')
plt.tight_layout()
plt.savefig('../plots/univariate_error_vs_epoch.png', dpi=150)
plt.close()
print("Saved error vs epoch plot")

# ---------- Step 7: Plot 2 - Model output vs Target (train and test) ----------
for name, Xr, yt, yp in [('train', X_train, y_train, y_pred_train),
                          ('test', X_test, y_test, y_pred_test)]:
    order = np.argsort(Xr[:, 0])   # x ke hisaab se sort karo taaki line clean dikhe
    plt.figure(figsize=(6, 4))
    plt.scatter(Xr[order, 0], yt[order], s=10, label='Target', c='blue')
    plt.scatter(Xr[order, 0], yp[order], s=10, label='Model Output', c='red', marker='x')
    plt.xlabel('x')
    plt.ylabel('y')
    plt.title(f'Univariate - Model vs Target ({name})')
    plt.legend()
    plt.tight_layout()
    plt.savefig(f'../plots/univariate_model_vs_target_{name}.png', dpi=150)
    plt.close()
    print(f"Saved model_vs_target_{name} plot")

# ---------- Step 8: Plot 3 - Scatter target vs predicted ----------
for name, yt, yp in [('train', y_train, y_pred_train), ('test', y_test, y_pred_test)]:
    plt.figure(figsize=(5, 5))
    plt.scatter(yt, yp, s=10)
    lims = [min(yt.min(), yp.min()), max(yt.max(), yp.max())]
    plt.plot(lims, lims, 'k--', label='Ideal (y=x)')
    plt.xlabel('Target Output')
    plt.ylabel('Model Output')
    plt.title(f'Univariate - Target vs Predicted ({name})')
    plt.legend()
    plt.tight_layout()
    plt.savefig(f'../plots/univariate_scatter_{name}.png', dpi=150)
    plt.close()
    print(f"Saved scatter_{name} plot")