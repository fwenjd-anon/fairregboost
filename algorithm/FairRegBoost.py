import sys
import copy
import itertools
import numpy as np
import pandas as pd
import xgboost as xgb
import ot
from imblearn.combine import SMOTETomek
from sklearn.model_selection import train_test_split
from scipy.spatial.distance import cdist
from scipy.stats import skew, kurtosis, wasserstein_distance, norm, normaltest
from sklearn.metrics import mean_squared_error
from sklearn.preprocessing import PolynomialFeatures, StandardScaler
from sklearn.feature_selection import SelectFromModel, mutual_info_regression
from sklearn.preprocessing import PowerTransformer
from sklearn.ensemble import IsolationForest

def loading_bar(iteration, total, length=30):
    percent = (iteration / total)
    bar_length = int(length * percent)
    bar = '█' * bar_length + '-' * (length - bar_length)
    sys.stdout.write(f'\r[{bar}] {percent*100:.2f}%')
    sys.stdout.flush()

class FairRegBoost:
    """This class calls the algorithm.

    Parameters
    ----------
    """
    def __init__(self, df_dict, preparation_nr, uc_strategy, lam, gamma):
        self.sensitive = df_dict["sens_attrs"]
        self.label = df_dict["label"]
        self.preparation_nr = preparation_nr
        self.uc_strategy = uc_strategy
        self.outlier, self.featineering, self.featselection, self.powertransform, self.balancing, self.augmentation = False, False, False, False, False, False
        self.alpha = 1
        self.lam = lam
        self.gamma = gamma

    def _evaluate_component(self, X_orig, y_orig, X, y, sensitive_attrs, label_attr, X_analysis, y_analysis, component, applied_components):
        """
        Evaluates the preparation component.
        """
        X_analysis2 = copy.deepcopy(X_analysis)
        X_analysis3 = copy.deepcopy(X_analysis)
        if component == "featineering":
            X_poly = self.poly.transform(X_analysis3)
            X_analysis3 = pd.DataFrame(X_poly, columns=self.poly.get_feature_names_out(X_analysis3.columns), index=y_analysis.index)
        if "Feature Generation" in applied_components:
            X_poly = self.poly.transform(X_analysis3)
            X_analysis3 = pd.DataFrame(X_poly, columns=self.poly.get_feature_names_out(X_analysis3.columns), index=y_analysis.index)
            X_analysis2 = copy.deepcopy(X_analysis3)

        if component == "featselection":
            X_analysis3 = X_analysis3[self.selected_features]
        if "Feature Selection" in applied_components:
            X_analysis2 = X_analysis2[self.selected_features]
            X_analysis3 = X_analysis3[self.selected_features]
        
        self.model.fit(X_orig, y_orig)
        pred = self.model.predict(X_analysis2)
        if "Power Transformation" in applied_components:
            pred = self.pt.inverse_transform(np.array(pred).reshape(-1, 1)).flatten()
        rmse_orig = mean_squared_error(y_analysis[label_attr], pred, squared=False)
        wasser = []
        X_analysis2["pred"] = pred
        X_analysis2[label_attr] = y_analysis[label_attr]
        gdf = X_analysis2.groupby(sensitive_attrs)
        for key, item in gdf:
            grp_df = gdf.get_group(key)
            wasser.append(abs(wasserstein_distance(grp_df["pred"], X_analysis2["pred"])))
        wd = sum(wasser)/(len(gdf)-1)
        result_orig = rmse_orig + wd

        
        self.model.fit(X, y)
        pred = self.model.predict(X_analysis3)
        if component == "powertransform" or "Power Transformation" in applied_components:
            pred = self.pt.inverse_transform(np.array(pred).reshape(-1, 1)).flatten()
        rmse = mean_squared_error(y_analysis[label_attr], pred, squared=False)
        wasser = []
        X_analysis3["pred"] = pred
        X_analysis3[label_attr] = y_analysis[label_attr]
        gdf = X_analysis3.groupby(sensitive_attrs)
        for key, item in gdf:
            grp_df = gdf.get_group(key)
            wasser.append(abs(wasserstein_distance(grp_df["pred"], X_analysis3["pred"])))
        wd = sum(wasser)/(len(gdf)-1)
        result = rmse + wd

        return result_orig > result


    def _iterative_fairness_component_selection(self, X, y, X_test, y_test, sensitive_attrs, label_attr):
        """
        Iteratively applies fairness-related data preparation components in an optimal sequence.
        """
        applied_components = []
        cols = X.columns
        X_orig = copy.deepcopy(X)
        y_orig = copy.deepcopy(y)

        '''
        # 1. Outlier Handling / old version
        if self.preparation_nr in (1, "auto"):
            print("Test for Outlier Handling")
            stat, p = normaltest(y)
            if p < 0.05 or self.preparation_nr==1:
                print("Applying Outlier Handling...")
                X = X.to_numpy()
                y = y.to_numpy()
                detector = IsolationForest(contamination=0.05)
                detector.fit(X)
                mask = detector.predict(X) == 1
                X = X[mask]
                y = y[mask]
                X = pd.DataFrame(X, columns=cols)
                y = pd.DataFrame(y, columns=[label_attr])
                if self.preparation_nr==1 or self._evaluate_component(X_orig.copy(), y_orig.copy(), X.copy(), y.copy(), sensitive_attrs, label_attr, X_test, y_test, "outlier", applied_components):
                    X_orig = copy.deepcopy(X)
                    y_orig = copy.deepcopy(y)
                    applied_components.append("Outlier Handling")
                    self.outlier = True
                else:
                    X = copy.deepcopy(X_orig)
                    y = copy.deepcopy(y_orig)
                    applied_components.append("Evaluated Outlier Handling")
        '''
        

        # 2. Feature Generation
        if self.preparation_nr in (2, "auto"):
            print("Test for Feature Generation")
            feature_cols = [col for col in X.columns if col not in sensitive_attrs]
            if len(feature_cols) < 50 or self.preparation_nr==2:
                feature_corrs = X[feature_cols].corrwith(y).abs()
                max_corr = feature_corrs.max() if not feature_corrs.empty else 0
                mutual_info = mutual_info_regression(X[feature_cols], y) if feature_cols else []
                if (max_corr < 0.5) or (len(mutual_info) > 0 and np.mean(mutual_info) > 0.1) or self.preparation_nr==2:
                    print("Applying Feature Generation...")
                    self.poly = PolynomialFeatures(degree=2, include_bias=False)
                    X_poly = self.poly.fit_transform(X)
                    X = pd.DataFrame(X_poly, columns=self.poly.get_feature_names_out(X.columns), index=y.index)
                    X.dropna(axis=1, inplace=True)
                    if self.preparation_nr==2 or self._evaluate_component(X_orig.copy(), y_orig.copy(), X.copy(), y.copy(), sensitive_attrs, label_attr, X_test, y_test, "featineering", applied_components):
                        X_orig = copy.deepcopy(X)
                        y_orig = copy.deepcopy(y)
                        applied_components.append("Feature Generation")
                        self.featineering = True
                    else:
                        X = copy.deepcopy(X_orig)
                        y = copy.deepcopy(y_orig)
                        applied_components.append("Evaluated Feature Generation")


        # 3. Featur Selection
        if self.preparation_nr in (3, "auto"):
            print("Test for Feature Selection")
            feature_cols = [col for col in X.columns if col not in sensitive_attrs]
            if len(feature_cols) >= 50 or self.preparation_nr==3:
                print("Applying Feature Selection...")
                mi_scores = mutual_info_regression(X, y, random_state=0)
                mi_df = pd.DataFrame({'feature': X.columns, 'mi_score': mi_scores})
                mi_df = mi_df.sort_values('mi_score', ascending=False).reset_index(drop=True)
                
                mi_df['cumulative_mi'] = mi_df['mi_score'].cumsum() / mi_df['mi_score'].sum()
                self.selected_features = mi_df[mi_df['cumulative_mi'] <= 0.9]['feature'].tolist()
                for sens in self.sensitive:
                    if sens not in self.selected_features:
                        self.selected_features.append(sens)
                X = X[self.selected_features]

                if self.preparation_nr==3 or self._evaluate_component(X_orig.copy(), y_orig.copy(), X.copy(), y.copy(), sensitive_attrs, label_attr, X_test, y_test, "featselection", applied_components):
                    X_orig = copy.deepcopy(X)
                    y_orig = copy.deepcopy(y)
                    applied_components.append("Feature Selection")
                    self.featselection = True
                else:
                    X = copy.deepcopy(X_orig)
                    y = copy.deepcopy(y_orig)
                    applied_components.append("Evaluated Feature Selection")

        '''
        #PowerTransformer
        if self.preparation_nr in (6, "auto"):
            print("Test for Power Transformation")
            label_skewness = abs(skew(y))
            label_kurtosis = kurtosis(y)
            if abs(label_skewness) > 1 or label_kurtosis > 3 or self.preparation_nr==6:
                print("Applying Power Transformation...")
                self.pt = PowerTransformer(method="yeo-johnson")
                transformed_series = self.pt.fit_transform(y[label_attr].values.reshape(-1, 1))
                y = pd.DataFrame(transformed_series.flatten(), columns=[label_attr], index=y.index)
                if self.preparation_nr==6 or self._evaluate_component(X_orig.copy(), y_orig.copy(), X.copy(), y.copy(), sensitive_attrs, label_attr, X_test, y_test, "powertransform", applied_components):
                    X_orig = copy.deepcopy(X)
                    y_orig = copy.deepcopy(y)
                    applied_components.append("Power Transformation")
                    self.powertransform = True
                else:
                    X = copy.deepcopy(X_orig)
                    y = copy.deepcopy(y_orig)
                    applied_components.append("Evaluated Power Transformation")
        '''

        # 4. Balancing
        if self.preparation_nr in (4, "auto"):
            print("Test for Balancing")
            X2 = copy.deepcopy(X)
            X2[label_attr] = y[label_attr]
            tuples = [tuple(row) for row in X2[self.sensitive].values]
            labels, uniques = pd.factorize(tuples)
            X2['group_label'] = labels
            df_map = pd.DataFrame(list(uniques), columns=self.sensitive)
            df_map.insert(0, 'label', range(len(df_map)))
            df_map.set_index('label', inplace=True)
            Y2 = X2['group_label']
            class_counts = Y2.value_counts(normalize=True) * 100
            min_class_percentage = class_counts.min()
            max_class_percentage = class_counts.max()
            if max_class_percentage/min_class_percentage > 1.5 or self.preparation_nr==4:
                print("Applying Balancing...")
                tl = SMOTETomek(random_state=42, n_jobs=-1)
                X, y = tl.fit_resample(X2, Y2)
                y = pd.DataFrame(y, columns=['group_label'])
                X['group_label'] = y['group_label']
                for sens in self.sensitive:
                    attr = []
                    for i, row in X.iterrows():
                        gl = int(row['group_label'])
                        attr.append(df_map.loc[gl, sens])
                    X[sens] = attr
                X.dropna(inplace=True)
                y = X[label_attr]
                y = pd.DataFrame(y, columns=[label_attr])
                X.drop(label_attr, axis=1, inplace=True)
                X.drop('group_label', axis=1, inplace=True)
                if self.preparation_nr==4 or self._evaluate_component(X_orig.copy(), y_orig.copy(), X.copy(), y.copy(), sensitive_attrs, label_attr, X_test, y_test, "balancing", applied_components):
                    X_orig = copy.deepcopy(X)
                    y_orig = copy.deepcopy(y)
                    applied_components.append("Balancing")
                    self.balancing = True
                else:
                    X = copy.deepcopy(X_orig)
                    y = copy.deepcopy(y_orig)
                    applied_components.append("Evaluated Balancing")

        # Step 5: Uncertainty-Aware Augmentation
        if self.preparation_nr in (5, "auto"):
            print("Test for Uncertainty-Aware Augmentation")
            self.model.fit(X, y)
            pred = self.model.predict(X)
            if self.powertransform:
                pred = self.pt.inverse_transform(np.array(pred).reshape(-1, 1)).flatten()
            residuals = (y[label_attr] - pred)**2
            X['residuals'] = residuals
            group_variance = X.groupby(sensitive_attrs)['residuals'].var()
            variance_std = group_variance.std() if not group_variance.empty else 0
            high_uncertainty = residuals > residuals.mean()
            X.drop("residuals", axis=1, inplace=True)
            if variance_std > 0.2 or self.preparation_nr==5:
                print("Applying Uncertainty-Aware Augmentation...")
                #noise = np.random.normal(0, uncertainty, size=len(X)).reshape(-1,1)
                X_aug = X.loc[high_uncertainty]
                noise_factor = 0.1
                noise = np.random.normal(0, noise_factor*X.std(), size=X_aug.shape)
                X_aug += noise
                y_aug = y.loc[high_uncertainty]
                X = pd.concat([X, X_aug], axis=0).reset_index(drop=True)
                y = pd.concat([y, y_aug], axis=0).reset_index(drop=True)
                try:
                    if self.preparation_nr==5 or self._evaluate_component(X_orig.copy(), y_orig.copy(), X.copy(), y.copy(), sensitive_attrs, label_attr, X_test, y_test, "augmentation", applied_components):
                        X_orig = copy.deepcopy(X)
                        y_orig = copy.deepcopy(y)
                        applied_components.append("Uncertainty-Aware Augmentation")
                        self.augmentation = True
                    else:
                        X = copy.deepcopy(X_orig)
                        y = copy.deepcopy(y_orig)
                        applied_components.append("Evaluated Uncertainty-Aware Augmentation")
                except:
                    X = copy.deepcopy(X_orig)
                    y = copy.deepcopy(y_orig)
                    applied_components.append("Evaluated Uncertainty-Aware Augmentation")

        return X, y, applied_components


    def fit(self, X, y):
        """
        """
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

        self.model = xgb.XGBRegressor(n_estimators=100, max_depth=3, learning_rate=0.1, subsample=0.8, colsample_bytree=0.8, random_state=42)
        
        # Do data preparation
        if self.preparation_nr != 0:
            df_train = copy.deepcopy(X_train)
            df_train[self.label] = y_train[self.label]
            X_train, y_train, self.components = self._iterative_fairness_component_selection(X_train.copy(), y_train.copy(), X_test.copy(), y_test.copy(), self.sensitive, self.label)
            df_train = copy.deepcopy(X_train)
            df_train[self.label] = y_train[self.label]

        # Train a regressor
        if self.featineering:
            X_poly = self.poly.transform(X_test)
            X_test = pd.DataFrame(X_poly, columns=self.poly.get_feature_names_out(X_test.columns), index=y_test.index)
        
        self.model.fit(X_train, y_train)
        pred = self.model.predict(X_test).reshape(-1,1)
        if self.powertransform:
            pred = self.pt.inverse_transform(np.array(pred).reshape(-1, 1)).flatten()
        self.cols = X_test.columns.tolist()
        
        # Determine prediction intervals using uncertainty
        residuals = y_test.to_numpy() - pred
        residual_std = np.std(residuals)
        z_score = norm.ppf(0.9)  # 90% confidence interval
        self.prediction_interval = z_score * residual_std

        # Train uncertainty model
        self.uncertainty_model = xgb.XGBClassifier()
        high_uncertainty = (np.abs(residuals) > residual_std).astype(int)  # Binary label for uncertainty
        self.uncertainty_model.fit(X_test, high_uncertainty)

        return self


    def predict(self, X_test):
        """
        """
        test_indices = X_test.index.tolist()

        X_test_orig = copy.deepcopy(X_test)
        if self.featineering:
            X_poly = self.poly.transform(X_test)
            X_test = pd.DataFrame(X_poly, columns=self.poly.get_feature_names_out(X_test.columns), index=X_test_orig.index)
        if self.featselection:
            X_test = X_test[self.selected_features]
        y_pred = self.model.predict(X_test)
        if self.powertransform:
            y_pred = self.pt.inverse_transform(np.array(y_pred).reshape(-1, 1)).flatten()
        uncertainty_scores = self.uncertainty_model.predict_proba(X_test)[:, 1]

        X_test_sample = copy.deepcopy(X_test)
        X_test_sample['pred'] = y_pred
        num_bins = 10
        X_test_sample['bin'] = pd.cut(X_test_sample['pred'], bins=num_bins, labels=False)
        subsample_size = 100
        data_subsample = X_test_sample.groupby('bin', group_keys=False).apply(
            lambda x: x.sample(frac=min(subsample_size / len(X_test_sample), 1.0), random_state=42)
        ).reset_index(drop=True)
        data_subsample = data_subsample.drop(columns=['bin', 'pred'])
        avg_preds = self.model.predict(data_subsample)
        if self.powertransform:
            avg_preds = self.pt.inverse_transform(np.array(avg_preds).reshape(-1, 1)).flatten()
        avg_uncertainty = self.uncertainty_model.predict_proba(data_subsample)[:, 1]

        combination_dict = {}
        c = 0
        for i, row in X_test[self.sensitive].iterrows():
            key = tuple(row)
            if key not in combination_dict:
                combination_dict[key] = []
            combination_dict[key].append(c)
            c += 1
            
        adjusted_preds = y_pred.copy()
        avg_weights = np.ones(len(avg_uncertainty)) / len(avg_uncertainty)
 
        final_preds = []
        indices = []
        for group_key, group in combination_dict.items():
            preds_full = y_pred[group]
            indices_full = X_test_orig.iloc[group].index.tolist()
            unc_scores_full = uncertainty_scores[group]
            Xt_full = X_test.iloc[group]

            num_batches = 1
            batch_size = len(preds_full)

            for i in range(num_batches):
                start_idx = i * batch_size
                end_idx = min((i + 1) * batch_size, len(preds_full))
                preds = preds_full[start_idx:end_idx]
                unc_scores = unc_scores_full[start_idx:end_idx]
                Xt = Xt_full.iloc[start_idx:end_idx]
                indices += indices_full[start_idx:end_idx]
                group_weights = np.ones(len(preds)) / len(preds)

                M = cdist(preds.reshape(-1, 1), avg_preds.reshape(-1, 1), metric='sqeuclidean')
                feature_dist = cdist(Xt, data_subsample, metric='cosine')
                uncertainty_matrix = np.tile(unc_scores[:, None], (1, len(avg_preds)))

                
                M = M / M.max()
                feature_dist = feature_dist / feature_dist.max()
                uncertainty_matrix = np.tile(unc_scores[:, None], (1, len(avg_preds)))
                uncertainty_matrix = uncertainty_matrix / uncertainty_matrix.max()
                if self.lam == "best":
                    k = 0.1
                    median_C2 = np.median(M)
                    S_median = np.median(feature_dist)
                    self.lam = k * (median_C2 / (S_median + 1e-8))
                if self.gamma == "best":
                    k = 0.1
                    median_C2 = np.median(M)
                    IQR_U = np.percentile(uncertainty_matrix, 75) - np.percentile(uncertainty_matrix, 25)
                    self.gamma = k * (median_C2 / (IQR_U + 1e-8))

                if self.uc_strategy == "high_uc":
                    M += self.gamma * (1 - uncertainty_matrix)
                elif self.uc_strategy == "low_uc":
                    M += self.gamma * uncertainty_matrix
                elif self.uc_strategy == "mid_uc":
                    uncs = unc_scores * (1 - unc_scores) * 2
                    uncertainty_matrix = np.tile(uncs[:, None], (1, len(avg_preds)))
                    uncertainty_matrix = uncertainty_matrix / uncertainty_matrix.max()
                    M += self.gamma * uncertainty_matrix

                M += self.lam * feature_dist
                M = M / M.max()

                if self.gamma == 0:
                    reg = 0.05
                else:
                    reg = 0.05 * np.mean(unc_scores)
                ot_plan = ot.sinkhorn(group_weights, avg_weights, M, reg)

                adj_preds = [
                    np.clip(
                        np.sum(ot_plan[i, :] * avg_preds) / np.sum(ot_plan[i, :]),
                        preds[i] - self.prediction_interval,
                        preds[i] + self.prediction_interval
                    )
                    if np.sum(ot_plan[i, :]) > 0 else preds[i]
                    for i in range(len(preds))
                ]

                for i in range(len(adj_preds)):
                    final_preds.append((self.alpha * adj_preds[i] + (1 - self.alpha) * preds[i]).item())                    

        tuples_dict = dict(list(zip(indices, final_preds)))
        y_pred_adjusted = [tuples_dict[id_] for id_ in test_indices]

        return y_pred_adjusted
