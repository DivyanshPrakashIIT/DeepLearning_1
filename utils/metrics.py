import numpy as np

def confusion_matrix(y_true, y_pred, labels):
    """
    labels: list jaise [1,2,3] — classes ke naam
    Returns matrix jaha row=actual, column=predicted
    """
    n = len(labels)
    label_to_idx = {lab: i for i, lab in enumerate(labels)}
    M = np.zeros((n, n), dtype=int)

    for actual, pred in zip(y_true, y_pred):
        i = label_to_idx[actual]
        j = label_to_idx[pred]
        M[i, j] += 1

    return M


def precision_recall_f1(cm):
    """cm = confusion matrix. Returns per-class precision, recall, f1 + means."""
    n = cm.shape[0]
    precision = np.zeros(n)
    recall = np.zeros(n)
    f1 = np.zeros(n)

    for c in range(n):
        TP = cm[c, c]
        FP = cm[:, c].sum() - TP     # column sum - TP
        FN = cm[c, :].sum() - TP     # row sum - TP

        precision[c] = TP / (TP + FP) if (TP + FP) > 0 else 0
        recall[c] = TP / (TP + FN) if (TP + FN) > 0 else 0
        f1[c] = (2 * precision[c] * recall[c] / (precision[c] + recall[c])
                 if (precision[c] + recall[c]) > 0 else 0)

    return precision, recall, f1


def accuracy(cm):
    return np.trace(cm) / cm.sum()   # diagonal sum / total sum