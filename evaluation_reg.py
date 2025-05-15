"""
This code evaluates the results of the experiments based on several metrics.

UPDATE FOR MULTIPLE GROUPS
"""
import warnings
import argparse
import ast
import copy
import shelve
import pandas as pd
import time
import math
import statistics
import numpy as np
import ot
from sklearn.neighbors import NearestNeighbors
from scipy.stats import iqr
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from scipy.stats import gaussian_kde, norm, wasserstein_distance, entropy, ks_2samp
from scipy.spatial.distance import cdist

warnings.simplefilter(action='ignore', category=FutureWarning)

def compute_lipschitz_fairness(y_pred, X2, protected_columns=None, metric='euclidean'):
    X = copy.deepcopy(X2)
    # Convert predictions to numpy array
    y_pred = y_pred.to_numpy()

    # Drop protected columns from the feature DataFrame
    if protected_columns is not None:
        X = X.drop(columns=protected_columns)
    
    # Convert DataFrame to numpy array for distance computation
    X = X.to_numpy()

    # Compute pairwise differences in predictions
    pred_diffs = np.abs(y_pred[:, None] - y_pred[None, :])  # |y_i - y_j|
    
    # Compute pairwise distances in feature space
    feature_diffs = cdist(X, X, metric=metric)  # ||x_i - x_j||
    
    # Avoid division by zero (self-distances)
    feature_diffs[feature_diffs == 0] = np.inf  # Set diagonal to inf
    
    # Compute ratios
    ratios = pred_diffs / feature_diffs  # |y_i - y_j| / ||x_i - x_j||
    
    # Return the maximum ratio (Lipschitz constant)
    return np.max(ratios)


def consistency_score(y_pred, X2, protected_columns=None, k=5) -> float:
    X = copy.deepcopy(X2)

    # Convert predictions to numpy array
    y_pred = y_pred.to_numpy()

    # Drop protected columns from the feature DataFrame
    if protected_columns is not None:
        X = X.drop(columns=protected_columns)
    
    # Convert DataFrame to numpy array for distance computation
    X = X.to_numpy()

    # Compute pairwise distances and find k-nearest neighbors
    nbrs = NearestNeighbors(n_neighbors=k+1, metric='euclidean').fit(X)
    distances, indices = nbrs.kneighbors(X)

    # Exclude self-comparison (first column is the point itself)
    neighbor_preds = y_pred[indices[:, 1:]]  # Shape: (n_samples, k)
    pred_diffs = np.abs(y_pred[:, np.newaxis] - neighbor_preds)  # Shape: (n_samples, k)

    # Compute average prediction difference for k-nearest neighbors
    avg_pred_diff = np.mean(pred_diffs, axis=1)

    # Return overall mean consistency score
    return np.mean(avg_pred_diff)


def demographic_parity_score(y_true, y_pred, sensitive_attribute):
    df = pd.DataFrame({'true': y_true, 'pred': y_pred, 'sensitive': sensitive_attribute})

    if np.isnan(df.values).any():
        return None
    
    # Unique classes and sensitive attribute values
    classes = df['pred'].unique()
    sensitive_values = df['sensitive'].unique()
    
    cls_probs = []
    for val in sensitive_values:
        subset = df[df['sensitive'] == val]
        prob = sum(subset['pred']) / len(subset)
        cls_probs.append(prob)
    dp_score = 0
    for cp in cls_probs:
        dp_score += abs(cp - statistics.mean(cls_probs))
    dp_score = dp_score / len(sensitive_values)
    
    return dp_score


def full_evaluation(link, input_file, index, sens_attrs, label, model_list):
    #Read the original dataset
    original_data = pd.read_csv("Datasets/" + input_file + ".csv", index_col=index)
    dataset = copy.deepcopy(original_data)
    original_data = original_data.drop(columns=[label])
    for sens in sens_attrs:
        original_data = original_data.loc[:, original_data.columns != sens]

    for i, model in enumerate(model_list):
        try:
            original_data_short = pd.read_csv(link + model + "_prediction.csv", index_col=index)
            if not original_data_short.isna().any().any():
                modelnr = i
                break
            else:
                pass
        except Exception as e:
            pass

    original_data_short = pd.merge(original_data_short, original_data, left_index=True, right_index=True)
    original_data_short = original_data_short.loc[:, original_data_short.columns != model_list[modelnr]]
    orig_datas = copy.deepcopy(original_data_short)
    original_data_short = original_data_short.loc[:, original_data_short.columns != label]
    valid_data = dataset.loc[orig_datas.index, :]

    for sens in sens_attrs:
        original_data_short = original_data_short.loc[:, original_data_short.columns != sens]
    dataset2 = copy.deepcopy(original_data_short)
    total_size = len(original_data_short)

    groups = dataset[sens_attrs].drop_duplicates(sens_attrs).reset_index(drop=True)
    actual_num_of_groups = len(groups)
    sensitive_groups = []
    sens_cols = groups.columns
    for i, row in groups.iterrows():
        sens_grp = []
        for col in sens_cols:
            sens_grp.append(row[col])
        sensitive_groups.append(tuple(sens_grp))

    df_count = 0
    result_df = pd.DataFrame(columns=["model", "score", "rmse", "w2", "rmse_disparity", "consistency"])
    cluster_res = pd.DataFrame()
    ccount = 0

    for model in model_list:
        result_df.at[df_count, "model"] = model
        try:
            df = pd.read_csv(link + model + "_prediction.csv", index_col=index)
            if np.isnan(df.values).any():
                continue
        except:
            continue

        df = pd.merge(df, original_data_short, left_index=True, right_index=True)
        orig_datas['pred']  = df[model]
        if actual_num_of_groups == 2:
            # Separate privileged and unprivileged groups
            privileged_df = orig_datas[orig_datas[sens_attrs[0]] == 0]
            unprivileged_df = orig_datas[orig_datas[sens_attrs[0]] == 1]

            # Compute RMSE disparity
            rmse_privileged = mean_squared_error(privileged_df[label], privileged_df['pred'], squared=False)
            rmse_unprivileged = mean_squared_error(unprivileged_df[label], unprivileged_df['pred'], squared=False)
            rmse_disparity = abs(rmse_privileged - rmse_unprivileged)

            # Compute MAE disparity
            mae_privileged = mean_absolute_error(privileged_df[label], privileged_df['pred'])
            mae_unprivileged = mean_absolute_error(unprivileged_df[label], unprivileged_df['pred'])
            mae_disparity = abs(mae_privileged - mae_unprivileged)

            # Compute R^2 score disparity
            r2_privileged = r2_score(privileged_df[label], privileged_df['pred'])
            r2_unprivileged = r2_score(unprivileged_df[label], unprivileged_df['pred'])
            r2_score_disparity = abs(r2_privileged - r2_unprivileged)

            # Compute mean difference
            mean_difference = abs(statistics.fmean(privileged_df['pred']) - statistics.fmean(unprivileged_df['pred']))

            # Compute median difference
            median_difference = abs(statistics.median(privileged_df['pred']) - statistics.median(unprivileged_df['pred']))

            dp = demographic_parity_score(orig_datas[label], orig_datas['pred'], orig_datas[sens_attrs[0]])
            wasserstein = wasserstein_distance(privileged_df['pred'], unprivileged_df['pred'])

            rmse = abs(mean_squared_error(orig_datas[label], orig_datas['pred'], squared=False))
            epsilon = 1e-8
            actual = np.where(orig_datas[label] == 0, epsilon, orig_datas[label])
            result_df.at[df_count, "rmse"] = rmse
            result_df.at[df_count, "wasserstein_distance"] = wasserstein

            result_df.at[df_count, "MdAPE"] = np.median(np.abs((actual - orig_datas['pred']) / actual))
            result_df.at[df_count, "mse"] = abs(mean_squared_error(orig_datas[label], orig_datas['pred']))
            
            result_df.at[df_count, "rmse_disparity"] = rmse_disparity
            result_df.at[df_count, "r2_score"] = r2_score(orig_datas[label], orig_datas['pred'])
            result_df.at[df_count, "r2_score_disparity"] = r2_score_disparity
            result_df.at[df_count, "mean_difference"] = mean_difference
            result_df.at[df_count, "median_difference"] = median_difference
            result_df.at[df_count, "demographic_parity"] = dp

            result_df.at[df_count, "kolmogorov_smirnoff"] = ks_2samp(privileged_df['pred'], unprivileged_df['pred'])[0]
            X = orig_datas.loc[:, orig_datas.columns != 'pred']
            X = X.loc[:, X.columns != label]
            y_pred = orig_datas['pred']
            result_df.at[df_count, "consistency"] = consistency_score(y_pred, X, sens_attrs)
            result_df.at[df_count, "lipschitz_euclidean"] = compute_lipschitz_fairness(y_pred, X, sens_attrs, 'euclidean')
            result_df.at[df_count, "lipschitz_cosine"] = compute_lipschitz_fairness(y_pred, X, sens_attrs, 'cosine')
            pred0 = unprivileged_df['pred'].to_numpy()
            pred1 = privileged_df['pred'].to_numpy()
            a = np.ones(len(pred0)) / len(pred0)
            b = np.ones(len(pred1)) / len(pred1)
            M = ot.dist(pred0.reshape(-1, 1), pred1.reshape(-1, 1), metric='euclidean')**2
            w2 = np.sqrt(ot.emd2(a, b, M))
            result_df.at[df_count, "w2"] = w2
            result_df.at[df_count, "score"] = 0.5*rmse+0.5*w2
            
            try:
                kde1 = gaussian_kde(privileged_df['pred'])
                kde2 = gaussian_kde(unprivileged_df['pred'])
                length = min(len(privileged_df), len(unprivileged_df))
                x_vals = np.linspace(min(privileged_df['pred'].min(), unprivileged_df['pred'].min()), max(privileged_df['pred'].max(), unprivileged_df['pred'].max()), length)
                kde1_vals = kde1(x_vals)
                kde2_vals = kde2(x_vals)
                result_df.at[df_count, "kl_divergence"] = entropy(kde1_vals + 1e-10, kde2_vals + 1e-10)
            except:
                pass

            labels = np.array(orig_datas[label])
            label_iqr = iqr(labels)
            label_mad = np.median(np.abs(labels - np.median(labels)))

            if label_iqr == 0:
                label_iqr = 1
            if label_mad == 0:
                label_mad = 1

            result_df.at[df_count, "rmse_iqr"] = rmse/label_iqr
            result_df.at[df_count, "wasserstein_distance_iqr"] = wasserstein/label_iqr
        else:
            rmse, mae, r2 = 0, 0, 0
            wasser = []
            w2 = []

            avg_rmse = mean_squared_error(orig_datas[label], orig_datas['pred'], squared=False)
            avg_mae = mean_absolute_error(orig_datas[label], orig_datas['pred'])
            avg_r2 = r2_score(orig_datas[label], orig_datas['pred'])

            gdf = orig_datas.groupby(sens_attrs)
            X = orig_datas.loc[:, orig_datas.columns != 'pred']
            X = X.loc[:, X.columns != label]
            y_pred = orig_datas['pred']
            pred0 = orig_datas['pred'].to_numpy()
            a = np.ones(len(pred0)) / len(pred0)
            keylist = []
            for key, item in gdf:
                keylist.append(key)
                grp_df = gdf.get_group(key)
                grp_rmse = mean_squared_error(grp_df[label], grp_df['pred'], squared=False)
                grp_mae = mean_absolute_error(grp_df[label], grp_df['pred'])
                grp_r2 = r2_score(grp_df[label], grp_df['pred'])

                rmse += abs(avg_rmse - grp_rmse)
                mae += abs(avg_mae - grp_mae)
                r2 += abs(avg_r2 - grp_r2)
                wasser.append(abs(wasserstein_distance(grp_df['pred'], orig_datas['pred'])))
                pred1 = grp_df['pred'].to_numpy()
                b = np.ones(len(pred1)) / len(pred1)
                M = abs(ot.dist(pred0.reshape(-1, 1), pred1.reshape(-1, 1), metric='euclidean')**2)
                w2.append(np.sqrt(ot.emd2(a, b, M)))

            result_df.at[df_count, "max_wasser"] = max(wasser)
            result_df.at[df_count, "rmse_disparity"] = rmse/(len(gdf)-1)
            result_df.at[df_count, "mae_disparity"] = mae/(len(gdf)-1)
            result_df.at[df_count, "r2_score_disparity"] = r2/(len(gdf)-1)
            result_df.at[df_count, "wasserstein_distance"] = sum(wasser)/(len(gdf)-1)
            w2 = sum(w2)/(len(gdf)-1)
            result_df.at[df_count, "w2"] = w2
            result_df.at[df_count, "max_w2"] = max(w2)
            result_df.at[df_count, "total_w2"] = str(w2)
            result_df.at[df_count, "total_w2"] = str(keylist)
            result_df.at[df_count, "consistency"] = consistency_score(y_pred, X, sens_attrs)

            result_df.at[df_count, "rmse"] = avg_rmse
            epsilon = 1e-8
            actual = np.where(orig_datas[label] == 0, epsilon, orig_datas[label])
            result_df.at[df_count, "MdAPE"] = np.median(np.abs((actual - orig_datas['pred']) / actual))

            result_df.at[df_count, "score"] = 0.5*avg_rmse + 0.5*w2

        df_count += 1

    return result_df


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--ds", type=str, help="Name of the input .csv file.")
    parser.add_argument("--folder", type=str, help="Directory of the generated output files.")
    parser.add_argument("--index", default="index", type=str, help="Column name containing the index\
        of each entry. Default given column name: index.")
    parser.add_argument("--sensitive", type=str, help="List of column names of the sensitive attributes.")
    parser.add_argument("--label", type=str, help="Column name of the target value.")
    parser.add_argument("--models", default=None, type=str, help="List of models.")
    parser.add_argument("--name", default="EVALUATION", type=str, help="Chosen evaluation file name.")
    args = parser.parse_args()

    input_file = args.ds
    link = args.folder
    index = args.index
    sens_attrs = ast.literal_eval(args.sensitive)
    label = args.label
    model_list = ast.literal_eval(args.models)
    name = args.name

    result_df = full_evaluation(link, input_file, index, sens_attrs, label, model_list)

    result_df.to_csv(link + name + "_" + str(input_file) + ".csv")