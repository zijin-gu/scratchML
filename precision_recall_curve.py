def precision_recall_curve(y_true, y_pred):
    """
    Compute the precision-recall curve.
    
    Args:
    - y_true: true binary labels
    - y_pred: predicted probabilities
    
    Returns:
    - precision: precision values for different probability thresholds
    - recall: recall values for different probability thresholds
    - thresholds: probability thresholds
    """
    # sort predictions in descending order
    sorted_indices = np.argsort(y_pred)[::-1]
    sorted_labels = y_true[sorted_indices]
    sorted_scores = y_pred[sorted_indices]
    
    # initialize variables
    tp = 0
    fp = 0
    fn = np.sum(sorted_labels == 1)
    precision = np.zeros_like(sorted_scores)
    recall = np.zeros_like(sorted_scores)
    thresholds = np.zeros_like(sorted_scores)

    # iterate through sorted predictions
    for i in range(len(sorted_scores)):
        if sorted_labels[i] == 1:
            tp += 1
            fn -= 1
        else:
            fp += 1

        precision[i] = tp / (tp + fp)
        recall[i] = tp / (tp + fn)
        thresholds[i] = sorted_scores[i]

    return precision, recall, thresholds
