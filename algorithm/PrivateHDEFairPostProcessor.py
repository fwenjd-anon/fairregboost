import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression, LogisticRegression
from algorithm.PPP_files.postprocess import *

class PrivateHDEFairPostProcessor():
    """
    information [...]

    References:
        ...
    """
    def __init__(self,
                 df_dict,
                 alpha,
                 bins,
                 eps):
        """
        Args:
        """
        self.df_dict = df_dict
        self.alpha = alpha
        self.bins = bins
        self.eps = eps
        self.sensitive = df_dict["sens_attrs"]


    def fit(self, X_train, y_train):
        """
        Information

        Args:

        Returns:
        """
        if len(self.sensitive) > 1:
            unique_combinations = X_train[self.sensitive].apply(tuple, axis=1).unique()
            self.combination_dict = {comb: idx for idx, comb in enumerate(unique_combinations)}
            X_train['sensitive_encoded'] = X_train[self.sensitive].apply(lambda x: self.combination_dict[tuple(x)], axis=1)
            #X_train.drop(self.sensitive, axis=1, inplace=True)
            S_train = X_train['sensitive_encoded']
            num_unique = X_train['sensitive_encoded'].nunique()
            #X_train.drop('sensitive_encoded', axis=1, inplace=True)
        else:
            S_train = X_train[self.sensitive[0]]
            num_unique = X_train[self.sensitive[0]].nunique()
            #X_train.drop(self.sensitive[0], axis=1, inplace=True)

        #self.scaler = StandardScaler()
        #y_scaled = self.scaler.fit_transform(y_train.values.reshape(-1, 1))
        #y_train = pd.Series(y_scaled.flatten(), index=y_train.index)

        self.reg = LinearRegression(fit_intercept=True)
        self.reg.fit(X_train.to_numpy(), y_train.to_numpy())

        if self.eps == "inf":
            self.eps = np.inf

        if self.bins == "best":
            self.bins = int(len(X_train) ** 0.25)
        elif self.bins == "auto":
            self.bins = int(np.sqrt(len(X_train)))

        self.model = PrivateHDEFairPostProcessor().fit(y_train.to_numpy().reshape(-1,), S_train.to_numpy().reshape(-1,), alpha=self.alpha, n_bins=self.bins, eps=self.eps)

        return self


    def predict(self, X_test):
        """
        Information

        Args:

        Returns:
        """
        if len(self.sensitive) > 1:
            X_test['sensitive_encoded'] = X_test[self.sensitive].apply(lambda x: self.combination_dict[tuple(x)], axis=1)
            S_test = X_test['sensitive_encoded']
            #X_test.drop(self.sensitive, axis=1, inplace=True)
            #X_test.drop('sensitive_encoded', axis=1, inplace=True)
        else:
            S_test = X_test[self.sensitive[0]]
            #X_test.drop(self.sensitive[0], axis=1, inplace=True)
        #pred_scaled = self.model.predict(X_test.to_numpy())
        #pred = self.scaler.inverse_transform(pred_scaled.reshape(-1, 1))[:,0]
        y_pred = self.reg.predict(X_test.to_numpy())
        pred = self.model.predict(y_pred.reshape(-1,), S_test.to_numpy().reshape(-1,))

        return pred
        