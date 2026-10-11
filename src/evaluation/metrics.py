import numpy as np
from sklearn.metrics import precision_score, recall_score, f1_score, balanced_accuracy_score, roc_auc_score, average_precision_score

def compute_classification_metrics(y_true, y_pred, y_scores=None):
    """Compute classification metrics for binary or multi-class classification."""
    metrics = {
        "precision": float(precision_score(y_true, y_pred, zero_division=0)),
        "recall": float(recall_score(y_true, y_pred, zero_division=0)),
        "f1": float(f1_score(y_true, y_pred, zero_division=0)),
        "balanced_accuracy": float(balanced_accuracy_score(y_true, y_pred))
    }
    if y_scores is not None:
        try:
            metrics["auroc"] = float(roc_auc_score(y_true, y_scores))
            metrics["auprc"] = float(average_precision_score(y_true, y_scores))
        except ValueError:
            metrics["auroc"] = 0.0
            metrics["auprc"] = 0.0
    return metrics

def compute_segmentation_metrics(mask_true, mask_pred, smooth=1e-6):
    """Compute pixel-level segmentation metrics: IoU, Dice, Recall, and False Positive Rate."""
    mask_true = mask_true.astype(bool)
    mask_pred = mask_pred.astype(bool)

    intersection = np.logical_and(mask_true, mask_pred).sum()
    union = np.logical_or(mask_true, mask_pred).sum()
    
    iou = (intersection + smooth) / (union + smooth)
    dice = (2.0 * intersection + smooth) / (mask_true.sum() + mask_pred.sum() + smooth)
    
    # Pixel recall and False Positive Rate (FPR)
    tp = intersection
    fn = np.logical_and(mask_true, np.logical_not(mask_pred)).sum()
    fp = np.logical_and(np.logical_not(mask_true), mask_pred).sum()
    tn = np.logical_and(np.logical_not(mask_true), np.logical_not(mask_pred)).sum()

    recall = tp / (tp + fn + smooth)
    fpr = fp / (fp + tn + smooth)

    return {
        "iou": float(iou),
        "dice": float(dice),
        "pixel_recall": float(recall),
        "false_positive_rate": float(fpr)
    }