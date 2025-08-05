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
    float: The R-squared score, which ranges from 0 to 1. A score of 1 indicates 
           perfect prediction, while a score of 0 indicates that the model does not 
           explain any of the variability of the response data around its mean.
    """
    y_true, y_pred = y_true.data, y_pred.data
    
    # Handle edge case
    if len(y_true) <= 1:
        return 1.0

    # Calculate mean once to avoid recalculation
    y_true_mean = xp.mean(y_true)
    
    # Compute numerator and denominator
    numerator = xp.sum(xp.square(y_true - y_pred))
    denominator = xp.sum(xp.square(y_true - y_true_mean))

    # Handle the case where the denominator is zero
    if denominator == 0.0:
        return 1.0 if numerator == 0.0 else 0.0
    
    return 1.0 - numerator / denominator


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
    y_true, y_pred = y_true.data, y_pred.data
    
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
    if positive_label not in all_labels:
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
    y_true, y_pred = y_true.data, y_pred.data
    unique_labels = xp.unique(y_true)
    if len(unique_labels) != 2:
        raise ValueError("Binary classification requires exactly 2 unique labels.")
    
    # Use the larger label as positive
    positive_label = xp.max(unique_labels)
    
    # Get confusion matrix: [[TN, FP], [FN, TP]]
    cm = confusion_matrix(y_true, y_pred, positive_label=positive_label)
    tn, fp, fn, tp = cm.ravel()
    
    # Calculate metrics with safe division
    total = tp + tn + fp + fn
    accuracy = (tp + tn) / total if total > 0 else 0.0
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0
    
    return accuracy, precision, recall, f1


def _multiclass_classification_metrics(y_true: Tensor, y_pred: Tensor):
    """
    Compute multiclass classification metrics using macro averaging.
    
    Parameters:
    y_true (array-like): True multiclass labels.
    y_pred (array-like): Predicted multiclass labels.
    
    Returns:
    tuple: (accuracy, precision, recall, f1)
    """
    y_true, y_pred = y_true.data, y_pred.data
    # Multiclass accuracy is simply the fraction of correct predictions
    accuracy = xp.sum(y_true == y_pred) / len(y_true)
    
    unique_labels = xp.unique(y_true)
    precisions, recalls, f1s = [], [], []
    
    # Compute per-class metrics
    for label in unique_labels:
        # Get confusion matrix for this class vs all others
        cm = confusion_matrix(y_true, y_pred, positive_label=label)
        tn, fp, fn, tp = cm.ravel()
        
        # Calculate per-class metrics
        class_precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        class_recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        class_f1 = 2 * class_precision * class_recall / (class_precision + class_recall) if (class_precision + class_recall) > 0 else 0.0
        
        precisions.append(class_precision)
        recalls.append(class_recall)
        f1s.append(class_f1)
    
    # Macro average across all classes
    precision = xp.mean(precisions)
    recall = xp.mean(recalls)
    f1 = xp.mean(f1s)
    
    return accuracy, precision, recall, f1


def compute_classification_metrics(y_true: Tensor, y_pred: Tensor):
    """
    Efficiently computes all classification metrics.
    
    Parameters:
    y_true (array-like): True labels.
    y_pred (array-like): Predicted labels.
    
    Returns:
    tuple: (accuracy, precision, recall, f1) metrics
    """
    y_true, y_pred = y_true.data, y_pred.data
    
    if len(y_true) != len(y_pred):
        raise ValueError("y_true and y_pred must have the same length.")
    
    unique_labels = xp.unique(y_true)
    
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
    y_true, y_pred = y_true.data, y_pred.data
    
    if len(y_true) != len(y_pred):
        raise ValueError("y_true and y_pred must have the same length.")
    
    return xp.sum(y_true == y_pred) / len(y_true)


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
