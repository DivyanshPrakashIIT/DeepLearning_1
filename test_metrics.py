import sys
sys.path.insert(0, '.')
import numpy as np
from utils.metrics import confusion_matrix, precision_recall_f1, accuracy

# Fake example: 3 classes, kuch sahi kuch galat predictions
y_true = [1,1,1,2,2,2,3,3,3]
y_pred = [1,1,2,2,2,2,3,3,1]   # thodi galtiyan jaan-bujh kar

cm = confusion_matrix(y_true, y_pred, labels=[1,2,3])
print("Confusion Matrix:\n", cm)

p, r, f1 = precision_recall_f1(cm)
print("Precision:", p)
print("Recall:", r)
print("F1:", f1)
print("Accuracy:", accuracy(cm))