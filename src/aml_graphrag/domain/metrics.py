"""Metrics for a rare positive class. ROC-AUC is reported but not trusted: with ~2% positives it looks good
even when the top of the ranking is mostly false alarms."""


def precision_at_k(y_true, scores, k):
    top = sorted(range(len(scores)), key=lambda i: -scores[i])[:k]
    return sum(y_true[i] == 1 for i in top) / k


def f1_at(y_true, scores, threshold):
    tp = sum(1 for t, s in zip(y_true, scores) if s >= threshold and t == 1)
    fp = sum(1 for t, s in zip(y_true, scores) if s >= threshold and t != 1)
    fn = sum(1 for t, s in zip(y_true, scores) if s < threshold and t == 1)
    precision = tp / (tp + fp) if tp + fp else 0.0
    recall = tp / (tp + fn) if tp + fn else 0.0
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
    return {"precision": precision, "recall": recall, "f1": f1}
