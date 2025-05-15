import copy
import math
import ast
import numpy as np
import pandas as pd
import xgboost as xgb
import statistics
import ot

class Wass2Geo:
    """This class implements Wasserstein-p geodesic adjustments for predictions."""

    def __init__(self, 
                 df_dict,
                 t=0.5,
                 bins="best"):
        """
        Args:
        """
        self.df_dict = df_dict
        self.t = t
        self.bins = bins
        self.sensitive = df_dict["sens_attrs"]
        

    def fit(self, X_train, y_train):
        """Train the model on the provided training data."""
        self.model = xgb.XGBRegressor()
        self.model.fit(X_train, y_train)
        return self

    def compute_quantiles_and_inverse(self, data, num_bins):
        """
        Compute quantiles and their inverse for a dataset.

        Parameters:
        data (np.array): Data for which quantiles are computed.
        num_bins (int): Number of bins for quantile computation.

        Returns:
        (np.array, np.array): Quantiles and their inverse.
        """
        quantiles = np.linspace(0, 1, num_bins + 1)
        q_values = np.quantile(data, quantiles)
        q_inverse = np.interp(data, q_values, quantiles)
        return q_values, q_inverse

    def geodesic_interpolation_inverse(self, q_da_inv, q_d_inv, t):
        """
        Perform interpolation between group inverse quantiles and barycenter inverse quantiles.

        Parameters:
        q_da_inv (np.array): Inverse quantiles for the group.
        q_d_inv (np.array): Inverse quantiles for the barycenter.
        t (float): Trade-off parameter.

        Returns:
        np.array: Interpolated inverse quantiles.
        """
        return (1 - t) * q_da_inv + t * q_d_inv

    def compute_exact_quantiles(self, data, num_bins):
        """
        Compute the exact quantile function q_{D_a}(i).

        Parameters:
        data (np.array): Group data for quantile computation.
        num_bins (int): Number of bins.

        Returns:
        np.array: Exact quantiles q_{D_a}(i).
        """
        N = len(data)
        sorted_data = np.sort(data)
        cdf_bins = np.linspace(0, 1, num_bins + 1)
        quantiles = []

        for i in range(len(cdf_bins)):
            # Compute the supremum as per the definition
            threshold = cdf_bins[i] * N
            quantiles.append(sorted_data[int(np.ceil(threshold)) - 1])


        return np.sort(np.array(quantiles))

    def compute_inverse_quantile(self, data, quantiles, num_bins):
        """
        Compute the inverse quantile function q_{D_a}^{-1}(s).

        Parameters:
        data (np.array): Group data.
        quantiles (np.array): Quantiles q_{D_a}(i).
        num_bins (int): Number of bins.

        Returns:
        np.array: Inverse quantile function q_{D_a}^{-1}(s).
        """
        sorted_data = np.sort(data)
        cdf_bins = np.linspace(0, 1, num_bins + 1)
        inverse_quantiles = []

        for s in cdf_bins:
            # Map s to the corresponding quantile
            inverse_quantiles.append(quantiles[int(np.ceil(s * num_bins)) - 1])

        return np.array(inverse_quantiles)

    def geodesic_interpolation_exact(self, q_da_inv, q_d_inv, t):
        """
        Perform geodesic interpolation between inverse quantiles.

        Parameters:
        q_da_inv (np.array): Inverse quantiles for group D_a.
        q_d_inv (np.array): Inverse quantiles for the barycenter.
        t (float): Trade-off parameter.

        Returns:
        np.array: Interpolated inverse quantiles q_{D_{a,t}}^{-1}(s).
        """
        return (1 - t) * q_da_inv + t * q_d_inv

    def adjust_predictions(self, predictions, quantiles, interpolated_quantiles):
        """
        Adjust predictions using the interpolated quantiles.

        Parameters:
        predictions (np.array): Original predictions for the group.
        quantiles (np.array): Quantiles q_{D_a}(i).
        interpolated_quantiles (np.array): Interpolated quantiles q_{D_{a,t}}(i).

        Returns:
        np.array: Adjusted predictions.
        """
        adjusted_preds = []
        for pred in predictions:
            # Handle predictions below the minimum quantile
            if pred < quantiles[0]:
                adjusted_preds.append(interpolated_quantiles[0])
            # Handle predictions above the maximum quantile
            elif pred > quantiles[-1]:
                adjusted_preds.append(interpolated_quantiles[-1])
            else:
                # Map to the interpolated quantiles within the range
                for i in range(len(quantiles) - 1):
                    if quantiles[i] <= pred <= quantiles[i + 1]:
                        # Linear interpolation within the bin
                        slope = (interpolated_quantiles[i + 1] - interpolated_quantiles[i]) / (quantiles[i + 1] - quantiles[i])
                        adjusted_pred = interpolated_quantiles[i] + slope * (pred - quantiles[i])
                        adjusted_preds.append(adjusted_pred)
                        break
        return np.array(adjusted_preds)


    def predict(self, X_test):
        """Adjust predictions using the Wasserstein-p geodesic method."""
        y_pred = self.model.predict(X_test)
        combination_dict = {}
        c = 0
        for i, row in X_test[self.sensitive].iterrows():
            key = tuple(row)
            if key not in combination_dict:
                combination_dict[key] = []
            combination_dict[key].append(c)
            c += 1

        group_quantiles = []
        group_inverse_quantiles = []
        group_weights = []
        indices = []

        if self.bins not in ("best", "bestV2"):
            num_bins = self.bins
        elif self.bins == "best":
            num_bins = int(np.sqrt(len(y_pred)))
        elif self.bins == "bestV2":
            num_bins = int(math.log2(len(y_pred)) + 1)

        # Compute exact quantiles and inverse quantiles for each group
        for group_key, group in combination_dict.items():
            group_preds = y_pred[group]
            indices += X_test.iloc[group].index.tolist()
            
            q_values = self.compute_exact_quantiles(group_preds, num_bins)
            q_inverse = self.compute_inverse_quantile(group_preds, q_values, num_bins)
            group_quantiles.append(q_values)
            group_inverse_quantiles.append(q_inverse)
            group_weights.append(len(group))

        # Compute the barycenter inverse quantile function
        group_weights = np.array(group_weights) / sum(group_weights)
        barycenter_inverse = np.average(group_inverse_quantiles, axis=0, weights=group_weights)

        # Adjust predictions for each group
        final_preds = []
        for group_key, group in combination_dict.items():
            group_preds = y_pred[group]
            q_values = self.compute_exact_quantiles(group_preds, num_bins)
            q_inverse = self.compute_inverse_quantile(group_preds, q_values, num_bins)

            # Geodesic interpolation of the inverse quantile function
            interpolated_inverse = self.geodesic_interpolation_exact(q_inverse, barycenter_inverse, self.t)

            # Map predictions through the interpolated quantile function
            adjusted_group_preds = self.adjust_predictions(group_preds, q_values, interpolated_inverse)
            final_preds.extend(adjusted_group_preds)

        # Map adjusted predictions back to the original indices
        tuples_dict = dict(list(zip(indices, final_preds)))
        y_pred_adjusted = [tuples_dict[id_] for id_ in X_test.index.tolist()]

        return y_pred_adjusted