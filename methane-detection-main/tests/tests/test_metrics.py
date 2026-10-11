import numpy as np
from src.evaluation.metrics import compute_classification_metrics, compute_segmentation_metrics

def test_classification_metrics():
    y_true = np.array([0, 1, 1, 0, 1])
    y_pred = np.array([0, 1, 0, 0, 1])
    metrics = compute_classification_metrics(y_true, y_pred)
    
    assert "precision" in metrics
    assert "recall" in metrics
    assert "f1" in metrics
    assert metrics["f1"] > 0.0

def test_segmentation_metrics():
    mask_true = np.array([[0, 1], [1, 1]])
    mask_pred = np.array([[0, 1], [0, 1]])
    metrics = compute_segmentation_metrics(mask_true, mask_pred)

    assert 0.0 <= metrics["iou"] <= 1.0
    assert 0.0 <= metrics["dice"] <= 1.0
    assert metrics["pixel_recall"] == 2.0 / 3.0