# Evaluation Plan and Metric Definitions

## 1. Classification Metrics
- **Precision**: Proportion of positive identifications that were actually correct.
- **Recall**: Proportion of actual positives identified correctly.
- **F1-Score**: Harmonic mean of precision and recall.
- **Balanced Accuracy**: Average of recall obtained on each class.
- **AUROC / AUPRC**: Threshold-independent metrics computed when score arrays are provided.

## 2. Segmentation Metrics
- **IoU (Intersection over Union)**: Overlap of predicted mask over ground truth mask.
- **Dice Coefficient**: Twice the overlap divided by the total number of pixels in both masks.
- **Pixel Recall & FPR**: Measure true positive detection versus false alarms.

## 3. Quantification (Disabled Phase 1)
- MAE, RMSE, MAPE, and Pearson correlation will remain disabled until verified emission rate ground truth and units are provided by Member 1.