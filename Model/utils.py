import numpy as np
from sklearn.metrics import precision_recall_curve, auc, recall_score, precision_score, f1_score

def calculate_metrics(all_preds, all_labels, is_prob_target=False):
    """Calculate comprehensive metrics for both fixed and optimal thresholds"""
    # For probability targets, we need to handle both raw probabilities and thresholded binary values
    binary_labels = (all_labels > 0.5).astype(int)
    binary_preds_fixed = (all_preds > 0.5).astype(int)

    # Fixed threshold (0.5) metrics
    fixed_tp = np.sum((binary_preds_fixed == 1) & (binary_labels == 1))
    fixed_tn = np.sum((binary_preds_fixed == 0) & (binary_labels == 0))
    fixed_fp = np.sum((binary_preds_fixed == 1) & (binary_labels == 0))
    fixed_fn = np.sum((binary_preds_fixed == 0) & (binary_labels == 1))

    fixed_fnr = fixed_fn / (binary_labels == 1).sum() if (binary_labels == 1).sum() > 0 else 0
    fixed_fpr = fixed_fp / (binary_labels == 0).sum() if (binary_labels == 0).sum() > 0 else 0
    fixed_recall = fixed_tp / (fixed_tp + fixed_fn) if (fixed_tp + fixed_fn) > 0 else 0
    fixed_precision = fixed_tp / (fixed_tp + fixed_fp) if (fixed_tp + fixed_fp) > 0 else 0
    fixed_f1 = 2 * (fixed_precision * fixed_recall) / (fixed_precision + fixed_recall) if (fixed_precision + fixed_recall) > 0 else 0
    fixed_accuracy = np.mean(binary_preds_fixed == binary_labels)

    # Calculate AUC-PR for fixed threshold using three points
    fixed_recall_list = [0, fixed_recall, 1]
    fixed_precision_list = [1, fixed_precision, 0]
    fixed_auc_pr = auc(fixed_recall_list, fixed_precision_list)

    # Optimal threshold metrics using binary labels
    precision, recall, thresholds = precision_recall_curve(binary_labels, all_preds)
    auc_pr = auc(recall, precision)
    f1_scores = 2 * (precision * recall) / (precision + recall + 1e-8)
    optimal_idx = np.argmax(f1_scores)
    optimal_threshold = thresholds[optimal_idx] if len(thresholds) > optimal_idx else 0.5
    opt_preds = (all_preds > optimal_threshold).astype(int)

    # Calculate optimal threshold metrics
    true_positives = np.sum((opt_preds == 1) & (binary_labels == 1))
    true_negatives = np.sum((opt_preds == 0) & (binary_labels == 0))
    false_positives = np.sum((opt_preds == 1) & (binary_labels == 0))
    false_negatives = np.sum((opt_preds == 0) & (binary_labels == 1))

    # Add prediction distribution analysis
    near_threshold_range = 0.05
    near_threshold_mask = (all_preds >= 0.5 - near_threshold_range) & (all_preds <= 0.5 + near_threshold_range)
    uncertain_ratio = np.mean(near_threshold_mask)
    uncertain_correct = np.mean(np.abs(all_preds[near_threshold_mask] - all_labels[near_threshold_mask]) < 0.5) if near_threshold_mask.any() else 0

    metrics = {
        'auc_pr': auc_pr,
        'fixed_auc_pr': fixed_auc_pr,
        'accuracy': np.mean(opt_preds == binary_labels),
        'recall': recall_score(binary_labels, opt_preds),
        'precision': precision_score(binary_labels, opt_preds),
        'f1': f1_score(binary_labels, opt_preds),
        'true_positives': true_positives,
        'true_negatives': true_negatives,
        'false_positives': false_positives,
        'false_negatives': false_negatives,
        'false_negative_rate': false_negatives / (binary_labels == 1).sum() if (binary_labels == 1).sum() > 0 else 0,
        'false_positive_rate': false_positives / (binary_labels == 0).sum() if (binary_labels == 0).sum() > 0 else 0,
        'optimal_threshold': optimal_threshold,
        'fixed_accuracy': fixed_accuracy,
        'fixed_recall': fixed_recall,
        'fixed_precision': fixed_precision,
        'fixed_f1': fixed_f1,
        'fixed_true_positives': fixed_tp,
        'fixed_true_negatives': fixed_tn,
        'fixed_false_positives': fixed_fp,
        'fixed_false_negatives': fixed_fn,
        'fixed_fnr': fixed_fnr,
        'fixed_fpr': fixed_fpr,
        'uncertain_ratio': uncertain_ratio,
        'uncertain_accuracy': uncertain_correct,
    }

    # Add probability-specific metrics only if using probability targets
    if is_prob_target:
        prob_metrics = {
            'mse': np.mean((all_preds - all_labels) ** 2),
            'mae': np.mean(np.abs(all_preds - all_labels)),
            'rmse': np.sqrt(np.mean((all_preds - all_labels) ** 2)),
            'mean_pred_prob': np.mean(all_preds),
            'mean_true_prob': np.mean(all_labels),
            'prob_correlation': np.corrcoef(all_preds, all_labels)[0, 1],
        }
        metrics.update(prob_metrics)

    return metrics
