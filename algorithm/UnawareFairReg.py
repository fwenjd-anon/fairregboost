import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from sklearn.linear_model import LinearRegression, LogisticRegression
from algorithm.UnawareFairReg_files.FairReg import FairReg
from algorithm.UnawareFairReg_files.data_prep import get_frequencies
from algorithm.UnawareFairReg_files.plots import plot_distributions_compare, plot_predictions_compare, plot_distributions, plot_predictions, plot_risk_history,plot_unfairness_history,plot_unfairness_vs_risk, plot_risk_unf_compare


class UnawareFairReg():
    """
    information [...]

    References:
        ...
    """
    def __init__(self,
                 df_dict,
                 base,
                 L,
                 eps):
        """
        Args:
        """
        self.df_dict = df_dict
        self.base = base
        self.L = L
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
            X_train.drop(self.sensitive, axis=1, inplace=True)
            S_train = X_train['sensitive_encoded']
            num_unique = X_train['sensitive_encoded'].nunique()
            X_train.drop('sensitive_encoded', axis=1, inplace=True)
        else:
            S_train = X_train[self.sensitive[0]]
            num_unique = X_train[self.sensitive[0]].nunique()
            X_train.drop(self.sensitive[0], axis=1, inplace=True)

        self.scaler = MinMaxScaler(feature_range=(-1, 1))
        y_scaled = self.scaler.fit_transform(y_train.values.reshape(-1, 1))
        y_train = pd.Series(y_scaled.flatten(), index=y_train.index)

        #y_train = y_train[self.df_dict["label"]]

        p = get_frequencies(S_train)

        X_train, X_unlab, S_train, S_unlab, y_train, y_unlab = train_test_split(X_train, S_train, y_train, train_size=0.5, stratify=S_train)

        self.reg = LinearRegression(fit_intercept=True)
        self.reg.fit(X_train.to_numpy(), y_train)

        clf = LogisticRegression()
        clf.fit(X_train.to_numpy(), S_train)

        T=10000

        if self.L == "best":
            self.L = int(len(X_train) ** 0.25)
        elif self.L == "auto":
            self.L = int(np.sqrt(len(X_train)))

        eps_list = [2**(-self.eps) for i in range(num_unique)]

        self.model = FairReg(self.reg, clf, B=1, K=num_unique,  p=p, eps=eps_list, T=T, keep_history = False)
        self.model.fit(X_unlab.to_numpy(), L=self.L, alg={'base':self.base, 'method':'ACSA2'})

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
            X_test.drop(self.sensitive, axis=1, inplace=True)
            X_test.drop('sensitive_encoded', axis=1, inplace=True)
        else:
            S_test = X_test[self.sensitive[0]]
            X_test.drop(self.sensitive[0], axis=1, inplace=True)
        pred_scaled = self.model.predict(X_test.to_numpy())
        pred = self.scaler.inverse_transform(pred_scaled.reshape(-1, 1))[:,0]
        #pred = self.model.predict(X_test.to_numpy())

        #plot_distributions_compare(self.model, self.reg, X_test, S_test)

        return pred
        