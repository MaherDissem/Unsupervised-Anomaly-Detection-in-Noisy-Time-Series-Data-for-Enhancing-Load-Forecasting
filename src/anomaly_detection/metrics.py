"""Anomaly metrics."""
import os
import numpy as np
from sklearn import metrics
import matplotlib.pyplot as plt


def compute_timeseriewise_retrieval_metrics(
    anomaly_prediction_weights, anomaly_ground_truth_labels, eval_plots_path, contam_rate=0.1, use_optimal_threshold=False
):
    """
    Computes retrieval statistics (AUROC, FPR, TPR).

    Args:
        anomaly_prediction_weights: [np.array or list] [N] Assignment weights
                                    per timeserie. Higher indicates higher
                                    probability of being an anomaly.
        anomaly_ground_truth_labels: [np.array or list] [N] Binary labels - 1
                                    if timeserie is an anomaly, 0 if not.
    """
    # we compute AUROC to evaluate the performance of the anomaly detection regardless of the selected threshold
    fpr, tpr, _ = metrics.roc_curve(
        anomaly_ground_truth_labels, anomaly_prediction_weights
    )
    auroc = metrics.roc_auc_score(
        anomaly_ground_truth_labels, anomaly_prediction_weights
    )
    draw_curve(fpr, tpr, auroc, eval_plots_path)

    # we select the anomaly score threshold as a percentile of the anomaly scores
    threshold = np.percentile(anomaly_prediction_weights, 100 - contam_rate * 100)
    anomaly_predicted_labels = (anomaly_prediction_weights > threshold).astype(int)

    f1 = metrics.f1_score(anomaly_ground_truth_labels, anomaly_predicted_labels)
    precision = metrics.precision_score(anomaly_ground_truth_labels, anomaly_predicted_labels)
    recall = metrics.recall_score(anomaly_ground_truth_labels, anomaly_predicted_labels)
    confusion_matrix = metrics.confusion_matrix(
        anomaly_ground_truth_labels, (anomaly_prediction_weights > threshold).astype(int)
    ).ravel()

    # for optimality analysis, we also compute the metrics for an optimal threshold (highest f1 score)
    if use_optimal_threshold:
        precision_list, recall_list, threshold_list = metrics.precision_recall_curve(
            anomaly_ground_truth_labels, anomaly_prediction_weights
        )
        f1_score_list = np.divide(
            2 * precision_list * recall_list,
            precision_list + recall_list,
            out=np.zeros_like(precision_list),
            where=(precision_list + recall_list) != 0,
        )
        k = f1_score_list.argmax()
    
        threshold = threshold_list[k]
        f1 = f1_score_list[k]
        precision = precision_list[k]
        recall = recall_list[k]
        confusion_matrix = metrics.confusion_matrix(
            anomaly_ground_truth_labels, (anomaly_prediction_weights > threshold).astype(int)
        ).ravel()
        # print(f"{k/len(thresholds)*100}% percentile of best threshold is the best thres: {threshold}")

    return {
        "auroc": auroc, 
        "fpr": fpr, 
        "tpr": tpr,
        
        "threshold": threshold,
        "f1": f1,
        "precision": precision,
        "recall": recall,
        "confusion_matrix": confusion_matrix, # tn, fp, fn, tp
    }


def compute_pointwise_retrieval_metrics(predicted_masks, ground_truth_masks, patch_size=8):
    """
    Computes point-wise statistics (AUROC, FPR, TPR) for anomaly segmentations
    and ground truth segmentation masks.

    Args:
        predicted_masks: [list of np.arrays or np.array] [N x L] Contains
                                generated segmentation masks.
        ground_truth_masks: [list of np.arrays or np.array] [N x L] Contains
                            predefined ground truth segmentation masks
    """
    if isinstance(predicted_masks, list):
        predicted_masks = np.stack(predicted_masks).squeeze(-1)
    if isinstance(ground_truth_masks, list):
        ground_truth_masks = np.stack(ground_truth_masks)

    n_seq, pred_seq_len = predicted_masks.shape[:2]
    n_seq, gt_seq_len = ground_truth_masks.shape[:2]
    pred_masks_patched = np.zeros((n_seq, pred_seq_len//patch_size))
    ground_truth_masks_patched = np.zeros((n_seq, gt_seq_len//patch_size))

    for i, (pred_mask, ground_truth_mask) in enumerate(zip(predicted_masks, ground_truth_masks)):
        patched_pred_mask = np.array([np.any(pred_mask[i:i+patch_size]) for i in range(0, len(pred_mask), patch_size)])
        patched_gt_mask = np.array([np.any(ground_truth_mask[i:i+patch_size]) for i in range(0, len(ground_truth_mask), patch_size)])
        pred_masks_patched[i] = patched_pred_mask
        ground_truth_masks_patched[i] = patched_gt_mask

    flat_predicted_masks = pred_masks_patched.ravel() # binary
    flat_ground_truth_masks_patched = ground_truth_masks_patched.ravel()

    f1 = metrics.f1_score(flat_ground_truth_masks_patched, flat_predicted_masks)
    precision = metrics.precision_score(flat_ground_truth_masks_patched, flat_predicted_masks)
    recall = metrics.recall_score(flat_ground_truth_masks_patched, flat_predicted_masks)
    confusion_matrix = metrics.confusion_matrix(
        flat_ground_truth_masks_patched, flat_predicted_masks
    ).ravel()

    return {
        "f1": f1,
        "precision": precision,
        "recall": recall,   
        "confusion_matrix": confusion_matrix,     
    }


def draw_curve(fpr, tpr, auroc, eval_plots_path):
    plt.plot(fpr, tpr, 'k--', label='ROC (area = {0:.4f})'.format(auroc), lw=2)

    plt.xlim([-0.05, 1.05])
    plt.ylim([-0.05, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('ROC Curve')
    plt.legend(loc="lower right")

    # error = 0.015
    # miss = 0.1
    # # plt.plot([error, error], [-0.05, 1.05], 'k:', lw=1)
    # # plt.plot([-0.05, 1.05], [1-miss, 1-miss], 'k:', lw=1)
    # error_y, miss_x = 0, 1
    # for i in range(len(fpr)):
    #     if fpr[i] <= error <= fpr[i + 1]:
    #         error_y = tpr[i]
    #     if tpr[i] <= 1-miss <= tpr[i + 1]:
    #         miss_x = fpr[i]
    # plt.scatter(error, error_y, c='k')
    # plt.scatter(miss_x, 1-miss, c='k')
    # plt.text(error, error_y, "({0}, {1:.4f})".format(error, error_y), color='k')
    # plt.text(miss_x, 1-miss, "({0:.4f}, {1})".format(miss_x, 1-miss), color='k')
    # plt.show()
    
    os.makedirs(eval_plots_path, exist_ok=True)
    plt.savefig(os.path.join(eval_plots_path, "roc_curve.png"))
