import numpy as np

HEALTHY = 3
EDEMA = 2
NET = 1
ET = 4

# Calculate nodewise Dice score for WT, CT, and ET for a single brain.
# Expects two 1D vectors of integers.
def calculate_node_dices(preds, labels):
    p, l = preds, labels

    wt_preds = np.where(p == HEALTHY, 0, 1)
    wt_labs = np.where(l == HEALTHY, 0, 1)
    wt_dice = calculate_dice_from_logical_array(wt_preds, wt_labs)

    ct_preds = np.isin(p, [NET, ET]).astype(int)
    ct_labs = np.isin(l, [NET, ET]).astype(int)
    ct_dice = calculate_dice_from_logical_array(ct_preds, ct_labs)

    at_preds = np.where(p == ET, 1, 0)
    at_labs = np.where(l == ET, 1, 0)
    at_dice = calculate_dice_from_logical_array(at_preds, at_labs)

    return [wt_dice, ct_dice, at_dice]

# Each tumor region (WT, CT, ET) is binarized for both the prediction and ground truth 
# and then the overlapping volume is calculated.
def calculate_dice_from_logical_array(binary_predictions, binary_ground_truth):
    true_positives = np.logical_and(binary_predictions == 1, binary_ground_truth == 1)
    false_positives = np.logical_and(binary_predictions == 1, binary_ground_truth == 0)
    false_negatives = np.logical_and(binary_predictions == 0, binary_ground_truth == 1)
    tp, fp, fn = np.count_nonzero(true_positives), np.count_nonzero(false_positives), np.count_nonzero(false_negatives)
    # The case where no such labels exist (only really relevant for ET case).
    if (tp + fp + fn) == 0:
        return 1
    return (2 * tp) / (2 * tp + fp + fn)


def compute_ASA(supervoxels, ground_truth):
    regions = [1, 2, 4]
    ASA_scores = {}

    for region in regions:
        # Get the supervoxels and ground truth for the current region
        region_supervoxels = (supervoxels == region)
        region_ground_truth = (ground_truth == region)

        # Compute the intersection of the supervoxels and ground truth
        intersection = np.logical_and(region_supervoxels, region_ground_truth)

        # Compute ASA for the current region
        numerator = np.sum(np.max(intersection, axis=(1,2,3)))
        denominator = np.sum(region_ground_truth)
        ASA = numerator / denominator

        ASA_scores[region] = ASA

    return ASA_scores
