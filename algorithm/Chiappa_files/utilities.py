import numpy as np
from collections import defaultdict

def compute_group_datasets(dataset, group_col, score_col):
    """
    Partition the dataset into groups based on group_col.
    """
    groups = defaultdict(list)
    for row in dataset:
        groups[row[group_col]].append(row[score_col])
    return groups

def compute_quantile(group_scores, i, B):
    """
    Compute the i-th quantile for a given group.
    """
    sorted_scores = np.sort(group_scores)
    N = len(sorted_scores)
    threshold = (i - 1) / B
    cumulative_sum = np.cumsum([1] * N) / N
    for idx, val in enumerate(cumulative_sum):
        if val > threshold:
            return sorted_scores[idx]
    return sorted_scores[-1]  # Max value as fallback

def inverse_quantile(group_scores, s, B):
    """
    Compute the inverse quantile for a given threshold s.
    """
    sorted_scores = np.sort(group_scores)
    for i, val in enumerate(sorted_scores):
        if val > s:
            return i
    return B  # Max bin as fallback

def interpolate_quantiles(q_group_inv, q_bary_inv, t, s, B):
    """
    Perform geodesic interpolation between group and barycenter quantiles.
    """
    interpolated_inv = (1 - t) * q_group_inv + t * q_bary_inv
    for i in range(1, B + 1):
        if interpolated_inv <= i:
            return s
    return 0

def wass_p_geodesic(dataset, group_col, score_col, B, t):
    """
    Main function to compute the Wass-p geodesic.
    """
    groups = compute_group_datasets(dataset, group_col, score_col)
    all_scores = np.array([row[score_col] for row in dataset])
    barycenter = all_scores

    results = {}
    for group, scores in groups.items():
        group_results = []
        for i in range(1, B + 1):
            # Compute group and barycenter quantiles
            q_group = compute_quantile(scores, i, B)
            q_bary = compute_quantile(barycenter, i, B)
            
            # Interpolate quantiles
            interpolated_q = (1 - t) * q_group + t * q_bary
            group_results.append(interpolated_q)
        results[group] = group_results

    return results
