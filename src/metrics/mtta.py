import numpy as np

def mean_time_to_accident(pred_seq, thresh=0.5):
    """
    pred_seq: list of (probabilities over time) arrays, each length T
    Returns mean index at which prob crosses thresh for positives; smaller = earlier.
    For smoke tests we treat each sample as if T steps; real impl will align to annotations.
    """
    times = []
    for p in pred_seq:
        p = np.asarray(p)
        idx = np.where(p >= thresh)[0]
        t = idx[0] if len(idx) else len(p)  # if never crosses, assign T (worst/late)
        times.append(t)
    return float(np.mean(times)) if times else float("nan")
