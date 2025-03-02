import torch
import torch.nn.functional as F

def weighted_asymmetric_focal_loss(y_pred, y_true, alpha=0.85, beta=7.0, gamma=2.5, is_prob_target=False):
    """
    Weighted Asymmetric Focal Loss with support for probability targets.
    
    Args:
        is_prob_target: If True, treats y_true as probabilities (0-1) rather than binary labels
    """
    if is_prob_target:
        # For probability targets, use a modified loss that handles continuous values
        bce_loss = - (alpha * y_true * torch.log(y_pred + 1e-8) * beta +
                     (1 - alpha) * (1 - y_true) * torch.log(1 - y_pred + 1e-8))
    else:
        # For binary targets, use the original loss
        bce_loss = - (alpha * y_true * torch.log(y_pred + 1e-8) * beta +
                     (1 - alpha) * (1 - y_true) * torch.log(1 - y_pred + 1e-8))
    
    # Compute probability of correct classification
    p_t = torch.exp(-bce_loss)
    
    # Enhanced focal weight
    focal_weight = (1 - p_t) ** gamma
    
    # Apply focal weight to loss
    loss = focal_weight * bce_loss
    return loss.mean()

def weighted_focal_loss(y_pred, y_true, gamma=2.0, base_pos_weight=2.0, fn_penalty=15.0, fp_penalty=8.0, is_prob_target=False):
    """
    Enhanced focal loss with better support for probability targets.
    
    Args:
        y_pred: Model predictions (logits)
        y_true: Target values (probabilities if is_prob_target=True, else binary)
        gamma: Focal loss power parameter
        base_pos_weight: Base weight for positive class
        fn_penalty: False negative penalty
        fp_penalty: False positive penalty
        is_prob_target: If True, treats y_true as probabilities (0-1)
    """
    y_pred = torch.sigmoid(y_pred)
    eps = 1e-7
    y_pred = y_pred.clamp(eps, 1 - eps)
    
    if is_prob_target:
        # For probability targets:
        # 1. Scale class weight continuously with the target probability
        class_weight = torch.where(y_true > 0, 
                                 base_pos_weight * (1 + y_true), # Higher weight for higher probabilities
                                 1.0 + (1 - y_true))  # Small weight for low probabilities
        
        # 2. Calculate focal weight based on prediction error
        pt = 1 - torch.abs(y_true - y_pred)  # Higher pt means better prediction
        focal_weight = (1 - pt) ** gamma
        
        # 3. Calculate MSE-like BCE loss
        bce_loss = - (y_true * torch.log(y_pred) + (1 - y_true) * torch.log(1 - y_pred))
        
        # 4. Error-specific weights based on prediction deviation
        error_magnitude = torch.abs(y_true - y_pred)
        fn_mask = (y_true > y_pred) & (error_magnitude > 0.2)  # Significant underestimation
        fp_mask = (y_pred > y_true) & (error_magnitude > 0.2)  # Significant overestimation
        
        error_weights = torch.ones_like(y_true.float())
        error_weights[fn_mask] = fn_penalty * error_magnitude[fn_mask]  # Scale penalty with error
        error_weights[fp_mask] = fp_penalty * error_magnitude[fp_mask]  # Scale penalty with error
        
    else:
        # Original binary target logic
        class_weight = torch.where(y_true == 1, base_pos_weight, 1.0)
        pt = torch.where(y_true == 1, y_pred, 1 - y_pred)
        focal_weight = (1 - pt) ** gamma
        bce_loss = - (y_true * torch.log(y_pred) + (1 - y_true) * torch.log(1 - y_pred))
        
        fn_mask = (y_true == 1) & (y_pred < 0.5)
        fp_mask = (y_true == 0) & (y_pred >= 0.5)
        
        error_weights = torch.ones_like(y_true.float())
        error_weights[fn_mask] = fn_penalty
        error_weights[fp_mask] = fp_penalty
    
    # Final weights
    final_weight = class_weight * focal_weight * error_weights
    
    # Reduce gamma to make loss less aggressive
    # Add confidence penalty to encourage spread out predictions
    confidence_penalty = -0.1 * (y_pred * torch.log(y_pred + 1e-7) + 
                                (1-y_pred) * torch.log(1-y_pred + 1e-7)).mean()
    
    return (final_weight * bce_loss).mean() + confidence_penalty

def improved_weighted_focal_loss_v1(y_pred, y_true, base_pos_weight=2.0, fn_penalty=15.0, fp_penalty=8.0, is_prob_target=False):
    """
    Enhanced version specifically targeting prediction clustering and FNR
    """
    y_pred = torch.sigmoid(y_pred)
    eps = 1e-7
    y_pred = y_pred.clamp(eps, 1 - eps)
    
    # 1. Base class weights to handle imbalance
    class_weight = torch.where(y_true > 0.5, 
                             base_pos_weight * torch.ones_like(y_true),
                             torch.ones_like(y_true))
    
    # 2. Anti-clustering term: penalize predictions near 0.5
    center_distance = torch.abs(y_pred - 0.5)
    clustering_penalty = torch.exp(-10 * center_distance) # High penalty near 0.5
    
    # 3. Enhanced FN penalty with confidence pushing
    fn_mask = (y_true > y_pred)
    fn_distance = y_true - y_pred
    fn_weight = fn_penalty * (1 + fn_distance)  # Reduced multiplier from 2.0 to 1.0
    
    # 4. Balanced FP penalty
    fp_mask = (y_pred > y_true)
    fp_distance = y_pred - y_true
    fp_weight = fp_penalty * (1 + fp_distance)
    
    # 4. More balanced confidence pushing
    confidence_push = torch.where(
        y_true > 0.5,
        torch.max(torch.zeros_like(y_true), 0.8 - y_pred),  # Reduced from 0.9
        torch.max(torch.zeros_like(y_true), y_pred - 0.2)   # Reduced from 0.3
    )
    
    # Base BCE loss
    bce_loss = F.binary_cross_entropy(y_pred, y_true, reduction='none')
    
    # Combine weights
    total_weight = class_weight.clone()
    total_weight[fn_mask] *= fn_weight[fn_mask]
    total_weight[fp_mask] *= fp_weight[fp_mask]
    
    # Reduced coefficients for auxiliary losses
    weighted_loss = total_weight * bce_loss
    confidence_loss = 5.0 * confidence_push.mean()  # Stronger confidence pushing
    anti_clustering_loss = 2.0 * clustering_penalty.mean()
    
    return weighted_loss.mean() + confidence_loss + anti_clustering_loss

def improved_weighted_focal_loss_v2(
    y_pred, y_true, 
    base_pos_weight=2.0, 
    fn_penalty=15.0, 
    fp_penalty=8.0, 
    is_prob_target=False, 
    target_type='basic',
    confidence_margin=0.1,
    confidence_weight=5.0
):
    """
    Enhanced version specifically targeting prediction clustering and FNR, with support for probability targets
    
    Args:
        target_type: 'basic', 'o1', 'o2', or 'o3' to specify the target criteria
    """
    y_pred = torch.sigmoid(y_pred)
    eps = 1e-7
    y_pred = y_pred.clamp(eps, 1 - eps)
    
    # Base BCE loss calculation
    bce_loss = F.binary_cross_entropy(y_pred, y_true, reduction='none')
    
    if is_prob_target:
        # Define positive criteria based on target type
        if target_type == 'basic':
            # targetY_prob: treat > 0 as positive
            positive_mask = y_true > 0
            threshold = 0.0
        elif target_type == 'o1':
            # targetY_o1_prob: treat > 0.5 as positive
            positive_mask = y_true > 0.5
            threshold = 0.5
        elif target_type == 'o2':
            # targetY_o2_prob: treat > 0.3 as positive
            positive_mask = y_true > 0.3
            threshold = 0.3
        elif target_type == 'o3':
            # targetY_o3_prob: treat > 0.7 as positive
            positive_mask = y_true > 0.7
            threshold = 0.7
        else:
            raise ValueError(f"Unknown target type: {target_type}")
        
        # For probability targets, error is based on continuous difference
        error = y_true - y_pred
        error_weights = torch.ones_like(y_true)
        
        # Penalize underestimation (when pred < target)
        fn_mask = (positive_mask & (y_pred < threshold))
        error_weights[fn_mask] += fn_penalty * error[fn_mask] * y_true[fn_mask]
        
        # Penalize overestimation (when pred > target)
        fp_mask = (~positive_mask & (y_pred > threshold))
        error_weights[fp_mask] += fp_penalty * (-error[fp_mask]) * (1 - y_true[fp_mask])
        
        # Confidence pushing for probability targets
        confidence_push = torch.where(
            positive_mask,
            torch.max(torch.zeros_like(y_true), threshold + confidence_margin - y_pred) * y_true,
            torch.max(torch.zeros_like(y_true), y_pred - (threshold - confidence_margin)) * (1 - y_true)
        )
        
    else:
        # For binary targets, error is based on threshold crossing
        error_weights = torch.ones_like(y_true)
        # False negatives: true is 1, pred is < 0.5
        fn_mask = (y_true > 0.5) & (y_pred < 0.5)
        error_weights[fn_mask] = fn_penalty
        # False positives: true is 0, pred is >= 0.5
        fp_mask = (y_true < 0.5) & (y_pred >= 0.5)
        error_weights[fp_mask] = fp_penalty
        
        # Confidence pushing for binary targets
        confidence_push = torch.where(
            y_true > 0.5,
            torch.max(torch.zeros_like(y_true), 0.8 - y_pred),
            torch.max(torch.zeros_like(y_true), y_pred - 0.2)
        )
    
    # Class weight calculation
    class_weight = base_pos_weight * y_true + (1 - y_true)
    
    # Anti-clustering penalty
    center_distance = torch.abs(y_pred - 0.5)
    clustering_penalty = torch.exp(-10 * center_distance)
    
    # Combine components
    total_weight = class_weight * error_weights
    weighted_loss = total_weight * bce_loss
    confidence_loss = confidence_weight * confidence_push.mean()
    anti_clustering_loss = 2.0 * clustering_penalty.mean()
    
    return weighted_loss.mean() + confidence_loss + anti_clustering_loss
