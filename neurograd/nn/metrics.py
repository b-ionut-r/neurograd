from neurograd import Tensor, xp


def r2_score(y_true: Tensor, y_pred: Tensor):
    """
    Calculate the R-squared (coefficient of determination) regression score.

    R-squared is a statistical measure that represents the proportion of the variance 
    for a dependent variable that's explained by an independent variable or variables 
    in a regression model. It provides an indication of goodness of fit and therefore 
    a measure of how well unseen samples are likely to be predicted by the model.

    Parameters:
    y_true (array-like): True values of the target variable.
    y_pred (array-like): Predicted values of the target variable.

    Returns:
    float: The R-squared score in (-inf, 1]. A score of 1 indicates
           perfect prediction; values <= 0 indicate poor fit.
    """
    # Accept Tensor or array-like
    if isinstance(y_true, Tensor):
        y_true = y_true.data
    if isinstance(y_pred, Tensor):
        y_pred = y_pred.data
    
    # Handle edge case
    if len(y_true) <= 1:
        return 1.0

    # Calculate mean once to avoid recalculation
    y_true_mean = xp.mean(y_true)
    
    # Compute numerator and denominator
    numerator = xp.sum(xp.square(y_true - y_pred))
    denominator = xp.sum(xp.square(y_true - y_true_mean))

    # Handle the case where the denominator is zero (support CuPy scalars)
    if float(denominator) == 0.0:
        return 1.0 if float(numerator) == 0.0 else 0.0
    
    return (1.0 - numerator / denominator).item()


def confusion_matrix(y_true: Tensor, y_pred: Tensor, positive_label=None):
    """
    Computes the confusion matrix for binary classification.

    Parameters:
    y_true (numpy.ndarray): Array of true labels.
    y_pred (numpy.ndarray): Array of predicted labels (discrete values).
    positive_label (int or str): The label representing the positive class.

    Returns:
    numpy.ndarray: Confusion matrix as a 2x2 array:
                    [[TN, FP],
                     [FN, TP]]
    """
    # Accept Tensor or array-like
    if isinstance(y_true, Tensor):
        y_true = y_true.data
    if isinstance(y_pred, Tensor):
        y_pred = y_pred.data
    
    # Determine positive label if not provided
    if positive_label is None:
        unique_labels = xp.unique(y_true)
        if len(unique_labels) == 2:
            # Choose the larger label as positive by default
            positive_label = xp.max(unique_labels)
        else:
            raise ValueError("For multiclass, please provide the positive label.")
    
    # Validate positive label exists in the data
    all_labels = xp.concatenate([y_true, y_pred])
    if not xp.any(all_labels == positive_label).item():
        raise ValueError(f"Positive label {positive_label} not found in data.")
    
    # Calculate confusion matrix components
    tp = xp.sum((y_pred == positive_label) & (y_true == positive_label))
    fp = xp.sum((y_pred == positive_label) & (y_true != positive_label))
    fn = xp.sum((y_pred != positive_label) & (y_true == positive_label))
    tn = xp.sum((y_pred != positive_label) & (y_true != positive_label))

    # Return in standard format: [[TN, FP], [FN, TP]]
    return xp.array([[tn, fp], [fn, tp]])


def _binary_classification_metrics(y_true: Tensor, y_pred: Tensor):
    """
    Compute binary classification metrics efficiently.
    
    Parameters:
    y_true (array-like): True binary labels.
    y_pred (array-like): Predicted binary labels.
    
    Returns:
    tuple: (accuracy, precision, recall, f1)
    """
    # Accept Tensor or array-like
    if isinstance(y_true, Tensor):
        y_true_data = y_true.data
    else:
        y_true_data = y_true
    if isinstance(y_pred, Tensor):
        y_pred_data = y_pred.data
    else:
        y_pred_data = y_pred

    unique_labels = xp.unique(y_true_data)
    if len(unique_labels) != 2:
        raise ValueError("Binary classification requires exactly 2 unique labels.")
    
    # Use the larger label as positive
    positive_label = xp.max(unique_labels)
    
    # Get confusion matrix: [[TN, FP], [FN, TP]]
    cm = confusion_matrix(y_true_data, y_pred_data, positive_label=positive_label)
    tn, fp, fn, tp = cm.ravel()
    
    # Calculate metrics with safe division
    total = (tp + tn + fp + fn).item()
    accuracy = (tp + tn) / total if total > 0 else 0.0
    denom_p = (tp + fp).item()
    precision = tp / denom_p if denom_p > 0 else 0.0
    denom_r = (tp + fn).item()
    recall = tp / denom_r if denom_r > 0 else 0.0
    pr_sum = (precision + recall)
    # precision/recall may be xp scalars or floats; handle both safely
    if hasattr(pr_sum, "item"):
        pr_sum_val = pr_sum.item()
    else:
        pr_sum_val = float(pr_sum)
    f1 = 2 * precision * recall / pr_sum if pr_sum_val > 0 else 0.0

    return accuracy.item(), precision.item(), recall.item(), f1.item()


def _multiclass_classification_metrics(y_true: Tensor, y_pred: Tensor):
    """
    Compute multiclass classification metrics using macro averaging.
    
    Parameters:
    y_true (array-like): True multiclass labels.
    y_pred (array-like): Predicted multiclass labels.
    
    Returns:
    tuple: (accuracy, precision, recall, f1)
    """
    # Accept Tensor or array-like
    if isinstance(y_true, Tensor):
        y_true_data = y_true.data
    else:
        y_true_data = y_true
    if isinstance(y_pred, Tensor):
        y_pred_data = y_pred.data
    else:
        y_pred_data = y_pred

    # Multiclass accuracy is simply the fraction of correct predictions
    accuracy = xp.sum(y_true_data == y_pred_data) / len(y_true_data)
    
    unique_labels = xp.unique(y_true_data)
    precisions, recalls, f1s = [], [], []
    
    # Compute per-class metrics
    for label in unique_labels:
        # Get confusion matrix for this class vs all others
        cm = confusion_matrix(y_true_data, y_pred_data, positive_label=label)
        tn, fp, fn, tp = cm.ravel()
        
        # Calculate per-class metrics
        denom_p = (tp + fp).item()
        class_precision = tp / denom_p if denom_p > 0 else 0.0
        denom_r = (tp + fn).item()
        class_recall = tp / denom_r if denom_r > 0 else 0.0
        pr_sum = class_precision + class_recall
        if hasattr(pr_sum, "item"):
            pr_sum_val = pr_sum.item()
        else:
            pr_sum_val = float(pr_sum)
        class_f1 = 2 * class_precision * class_recall / pr_sum if pr_sum_val > 0 else 0.0
        
        precisions.append(class_precision)
        recalls.append(class_recall)
        f1s.append(class_f1)
    
    # Macro average across all classes
    precision = xp.mean(precisions)
    recall = xp.mean(recalls)
    f1 = xp.mean(f1s)

    return accuracy.item(), precision.item(), recall.item(), f1.item()


def compute_classification_metrics(y_true: Tensor, y_pred: Tensor):
    """
    Efficiently computes all classification metrics.
    
    Parameters:
    y_true (array-like): True labels.
    y_pred (array-like): Predicted labels.
    
    Returns:
    tuple: (accuracy, precision, recall, f1) metrics
    """
    if len(y_true) != len(y_pred):
        raise ValueError("y_true and y_pred must have the same length.")
    
    # Determine number of classes from true labels
    true_labels = y_true.data if isinstance(y_true, Tensor) else y_true
    unique_labels = xp.unique(true_labels)
    
    if len(unique_labels) == 2:
        return _binary_classification_metrics(y_true, y_pred)
    else:
        return _multiclass_classification_metrics(y_true, y_pred)


def accuracy_score(y_true: Tensor, y_pred: Tensor):
    """
    Computes the accuracy score for binary or multi-class classification.

    Accuracy is the fraction of predictions that match the true labels:
        accuracy = (correct_predictions) / (total_predictions)

    Parameters:
    y_true (array-like): Array of true labels.
    y_pred (array-like): Array of predicted labels.

    Returns:
    float: Accuracy score between 0.0 and 1.0.
    """
    # Accept Tensor or array-like
    if isinstance(y_true, Tensor):
        y_true = y_true.data
    if isinstance(y_pred, Tensor):
        y_pred = y_pred.data
    
    if len(y_true) != len(y_pred):
        raise ValueError("y_true and y_pred must have the same length.")

    return (xp.sum(y_true == y_pred) / len(y_true)).item()


def precision_score(y_true: Tensor, y_pred: Tensor, average='macro'):
    """
    Calculate the precision score for binary or multiclass classification.
    
    Parameters:
    y_true (array-like): True labels.
    y_pred (array-like): Predicted labels.
    average (str): Averaging strategy for multiclass ('macro' or 'micro').
    
    Returns:
    float: Precision score.
    """
    y_true, y_pred = y_true.data, y_pred.data
    
    unique_labels = xp.unique(y_true)
    
    if len(unique_labels) == 2:
        # Binary classification
        _, precision, _, _ = _binary_classification_metrics(y_true, y_pred)
        return precision
    else:
        # Multiclass classification
        if average == 'macro':
            _, precision, _, _ = _multiclass_classification_metrics(y_true, y_pred)
            return precision
        elif average == 'micro':
            # Micro-averaging: calculate metrics globally
            return accuracy_score(y_true, y_pred)  # For precision, micro-avg equals accuracy
        else:
            raise ValueError("average must be 'macro' or 'micro'")


def recall_score(y_true: Tensor, y_pred: Tensor, average='macro'):
    """
    Calculate the recall score for binary or multiclass classification.

    Parameters:
    y_true (array-like): True labels.
    y_pred (array-like): Predicted labels.
    average (str): Averaging strategy for multiclass ('macro' or 'micro').

    Returns:
    float: Recall score.
    """
    y_true, y_pred = y_true.data, y_pred.data
    unique_labels = xp.unique(y_true)
    
    if len(unique_labels) == 2:
        # Binary classification
        _, _, recall, _ = _binary_classification_metrics(y_true, y_pred)
        return recall
    else:
        # Multiclass classification
        if average == 'macro':
            _, _, recall, _ = _multiclass_classification_metrics(y_true, y_pred)
            return recall
        elif average == 'micro':
            # Micro-averaging: calculate metrics globally
            return accuracy_score(y_true, y_pred)  # For recall, micro-avg equals accuracy
        else:
            raise ValueError("average must be 'macro' or 'micro'")


def f1_score(y_true: Tensor, y_pred: Tensor, average='macro'):
    """
    Calculate the F1 score, which is the harmonic mean of precision and recall.
    
    Parameters:
    y_true (array-like): True labels.
    y_pred (array-like): Predicted labels.
    average (str): Averaging strategy for multiclass ('macro' or 'micro').
    
    Returns:
    float: F1 score.
    """
    y_true, y_pred = y_true.data, y_pred.data
    
    unique_labels = xp.unique(y_true)
    
    if len(unique_labels) == 2:
        # Binary classification
        _, _, _, f1 = _binary_classification_metrics(y_true, y_pred)
        return f1
    else:
        # Multiclass classification
        if average == 'macro':
            _, _, _, f1 = _multiclass_classification_metrics(y_true, y_pred)
            return f1
        elif average == 'micro':
            # Micro-averaging: F1 equals accuracy for multiclass
            return accuracy_score(y_true, y_pred)
        else:
            raise ValueError("average must be 'macro' or 'micro'")


def top_k_accuracy_score(y_true: Tensor, y_pred_scores: Tensor, k: int = 1):
    """
    Compute Top-k accuracy for multi-class predictions.

    Expects `y_pred_scores` of shape (N, C) with class scores/logits and
    `y_true` of shape (N,) with integer class indices. Counts a prediction as
    correct if the true label is among the top-k scores.

    Parameters:
    y_true: True class indices (Tensor or array-like), shape (N,)
    y_pred_scores: Predicted scores/logits (Tensor or array-like), shape (N, C)
    k: Top-k (e.g., 1, 3, 5)

    Returns:
    float: Top-k accuracy between 0.0 and 1.0
    """
    # Minimal conversions, following template of other metrics
    if isinstance(y_true, Tensor):
        y_true = y_true.data
    if isinstance(y_pred_scores, Tensor):
        y_pred_scores = y_pred_scores.data

    # Clamp k to number of classes
    num_classes = y_pred_scores.shape[1]
    k = int(k)
    if k < 1:
        k = 1
    if k > num_classes:
        k = num_classes

    # Get top-k indices per row using argpartition (no need to sort fully)
    # Use negative scores to select largest values
    topk_idx = xp.argpartition(-y_pred_scores, kth=k - 1, axis=1)[:, :k]

    # Compare true labels with top-k predictions
    correct = xp.any(topk_idx == y_true.reshape(-1, 1), axis=1)
    return xp.mean(correct).item()


def top_k_accuracies(y_true: Tensor, y_pred_scores: Tensor, ks=(1, 3, 5)):
    """
    Convenience helper to compute multiple Top-k accuracies at once.

    Parameters:
    y_true: True class indices (Tensor or array-like), shape (N,)
    y_pred_scores: Predicted scores/logits (Tensor or array-like), shape (N, C)
    ks: Iterable of k values (e.g., (1, 3, 5))

    Returns:
    list[float]: Top-k accuracies corresponding to each k in `ks`.
    """
    return [top_k_accuracy_score(y_true, y_pred_scores, int(k)) for k in ks]
