import numpy as np

def first_crossing_index(prob_seq, thresh=0.5):
    """Return first index where prob >= thresh, or len(seq) if never crosses."""
    p = np.asarray(prob_seq, dtype=np.float32)
    idx = np.where(p >= thresh)[0]
    return int(idx[0]) if len(idx) else len(p)

def relative_mtta_frames(prob_seq, thresh=0.5):
    """
    Relative mTTA in frames = (len(seq) - first_crossing_index).
    Larger is better (earlier alarm). 0 means we only crossed at the last frame; 
    if never crossed, returns 0.
    """
    L = len(prob_seq)
    t = first_crossing_index(prob_seq, thresh=thresh)
    return max(0, L - t)

def mean_relative_mtta(seqs, thresh=0.5):
    """
    seqs: list of probability sequences (positives only)
    Returns mean relative mTTA in frames.
    """
    if not seqs:
        return float("nan")
    vals = [relative_mtta_frames(s, thresh=thresh) for s in seqs]
    return float(np.mean(vals))

def frames_to_seconds(frames, fps=30.0):
    return float(frames) / float(fps)
