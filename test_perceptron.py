import sys
sys.path.insert(0, '.')
import numpy as np
from utils.perceptron import Perceptron

# Chota fake dataset — 2 clusters
np.random.seed(0)
X1 = np.random.randn(50, 2) + np.array([2, 2])
X2 = np.random.randn(50, 2) + np.array([-2, -2])
X = np.vstack([X1, X2])
t = np.array([1]*50 + [0]*50)   # sigmoid ke liye target 0/1

model = Perceptron(activation='sigmoid', lr=0.1, epochs=500)
model.fit(X, t)

predictions = (model.predict_raw(X) >= 0.5).astype(int)
accuracy = (predictions == t).mean()
print("Accuracy:", accuracy)
print("Final error:", model.error_history[-1])

# Tanh activation test
t2 = np.array([1]*50 + [-1]*50)   # tanh ke liye target -1/1 hota hai, 0/1 nahi
model2 = Perceptron(activation='tanh', lr=0.1, epochs=500)
model2.fit(X, t2)

predictions2 = np.where(model2.predict_raw(X) >= 0, 1, -1)
accuracy2 = (predictions2 == t2).mean()
print("Tanh Accuracy:", accuracy2)
print("Tanh Final error:", model2.error_history[-1])