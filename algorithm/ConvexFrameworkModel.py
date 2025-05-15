from algorithm.FGLM_files.models import ConvexFrameworkModel

class ConvexFrameworkModelClass():
    """
    information [...]

    References:
        ...
    """
    def __init__(self,
                 df_dict,
                 lam,
                 penalty):
        """
        Args:
        """
        self.df_dict = df_dict
        self.lam = lam
        self.penalty = penalty
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
            sens_idx = X_train.columns.get_loc('sensitive_encoded')
        else:
            sens_idx = X_train.columns.get_loc(self.sensitive[0])

        self.model = ConvexFrameworkModel(sensitive_index=sens_idx, sensitive_predictor=False, lam=self.lam, family="normal", penalty=self.penalty, tol=1e-4)
        self.model.fit(X_train.to_numpy(), y_train.to_numpy().reshape((y_train.to_numpy().shape[0],)))

        return self


    def predict(self, X_test):
        """
        Information

        Args:

        Returns:
        """
        if len(self.sensitive) > 1:
            X_test['sensitive_encoded'] = X_test[self.sensitive].apply(lambda x: self.combination_dict[tuple(x)], axis=1)
            X_test.drop(self.sensitive, axis=1, inplace=True)
        pred = self.model._predict(X_test.to_numpy())

        return pred
        