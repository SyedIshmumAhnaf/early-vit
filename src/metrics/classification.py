import numpy as np
from sklearn.metrics import average_precision_score, roc_auc_score

def ap_auc(preds, labels):
    """
    preds: (N,) sigmoid probabilities
    labels: (N,) {0,1}
    Returns AP, AUC (NaN-safe)
    """
    y = np.asarray(labels).astype(np.int32)
    p = np.asarray(preds).astype(np.float32)
    ap = average_precision_score(y, p) if len(np.unique(y)) > 1 else float("nan")
    try:
        auc = roc_auc_score(y, p) if len(np.unique(y)) > 1 else float("nan")
    except ValueError:
        auc = float("nan")
    return ap, auc
