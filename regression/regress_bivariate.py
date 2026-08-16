import sys
sys.path.insert(0, '..')
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D  # 3D plot ke liye zaroori
import os

from utils.perceptron import Perceptron

os.makedirs('../plots', exist_ok=True)

def rmse(y_true, y_pred):
    return np.sqrt(np.mean((y_true - y_pred) ** 2))

def percent_rmse(y_true, y_pred):
    return 100 * rmse(y_true, y_pred) / np.mean(np.abs(y_true))

# ---------- Step 1: Data load ----------
data = np.loadtxt('../Group19/Group19/Regression/BivariateData/19.csv', delimiter=',')
X = data[:, [0, 1]]   # x1, x2
y = data[:, 2]

print("X shape:", X.shape, "y shape:", y.shape)

# ---------- Step 2: Split ----------
np.random.seed(1)
n = X.shape[0]
idx = np.random.permutation(n)
n_train = int(0.7 * n)
train_idx, test_idx = idx[:n_train], idx[n_train:]

X_train, y_train = X[train_idx], y[train_idx]
X_test, y_test = X[test_idx], y[test_idx]
print("Train:", X_train.shape, "Test:", X_test.shape)

# ---------- Step 3: Standardize ----------
mu, sigma = X_train.mean(axis=0), X_train.std(axis=0)
X_train_s = (X_train - mu) / sigma
X_test_s = (X_test - mu) / sigma

# ---------- Step 4: Train ----------
model = Perceptron(activation='linear', lr=0.05, epochs=1000, seed=1)
model.fit(X_train_s, y_train)

y_pred_train = model.predict_raw(X_train_s)
y_pred_test = model.predict_raw(X_test_s)

# ---------- Step 5: RMSE ----------
train_rmse = rmse(y_train, y_pred_train)
test_rmse = rmse(y_test, y_pred_test)
train_prmse = percent_rmse(y_train, y_pred_train)
test_prmse = percent_rmse(y_test, y_pred_test)

print(f"Train RMSE: {train_rmse:.4f}  | %RMSE: {train_prmse:.2f}%")
print(f"Test RMSE:  {test_rmse:.4f}  | %RMSE: {test_prmse:.2f}%")

# ---------- Step 6: Error vs Epoch ----------
plt.figure(figsize=(6, 4))
plt.plot(model.error_history)
plt.xlabel('Epoch'); plt.ylabel('Average Error')
plt.title('Bivariate Regression - Error vs Epoch')
plt.tight_layout()
plt.savefig('../plots/bivariate_error_vs_epoch.png', dpi=150)
plt.close()

# ---------- Step 7: 3D Model vs Target ----------
for name, Xr, yt, yp in [('train', X_train, y_train, y_pred_train),
                          ('test', X_test, y_test, y_pred_test)]:
    fig = plt.figure(figsize=(6, 5))
    ax = fig.add_subplot(111, projection='3d')
    ax.scatter(Xr[:, 0], Xr[:, 1], yt, s=8, c='blue', label='Target')
    ax.scatter(Xr[:, 0], Xr[:, 1], yp, s=8, c='red', marker='^', label='Model Output')
    ax.set_xlabel('x1'); ax.set_ylabel('x2'); ax.set_zlabel('y')
    ax.set_title(f'Bivariate - Model vs Target ({name})')
    ax.legend()
    plt.tight_layout()
    plt.savefig(f'../plots/bivariate_model_vs_target_{name}.png', dpi=150)
    plt.close()
    print(f"Saved 3D plot ({name})")

# ---------- Step 8: Scatter target vs predicted ----------
for name, yt, yp in [('train', y_train, y_pred_train), ('test', y_test, y_pred_test)]:
    plt.figure(figsize=(5, 5))
    plt.scatter(yt, yp, s=10)
    lims = [min(yt.min(), yp.min()), max(yt.max(), yp.max())]
    plt.plot(lims, lims, 'k--', label='Ideal (y=x)')
    plt.xlabel('Target Output'); plt.ylabel('Model Output')
    plt.title(f'Bivariate - Target vs Predicted ({name})')
    plt.legend()
    plt.tight_layout()
    plt.savefig(f'../plots/bivariate_scatter_{name}.png', dpi=150)
    plt.close()
    print(f"Saved scatter plot ({name})")