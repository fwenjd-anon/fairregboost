import copy
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from algorithm.NIPS_files.utilities import *

class FairWassBary():
    """
    information [...]

    References:
        ...
    """
    def __init__(self,
                 df_dict):
        """
        Args:
        """
        self.df_dict = df_dict


    def fit(self, X_train, y_train):
        """
        Information

        Args:

        Returns:
        """
        self.A_train = X_train[self.df_dict["sens_attrs"][0]].values
        X_no_A = X_train.drop(self.df_dict["sens_attrs"], axis=1).values

        d = X_train.shape[1]

        # Train Random Forest
        self.classifier = RandomForestRegressor(n_estimators=500, random_state=13)
        self.classifier.fit(X_train.values, y_train.values)

        self.y_train_pred = self.classifier.predict(X_train.values)

        return self


    def predict(self, X_test):
        """
        Information

        Args:

        Returns:
        """
        A_test = X_test[self.df_dict["sens_attrs"][0]].values
        y_test_pred = self.classifier.predict(X_test.values)

        pred = f_NIPS(self.y_train_pred + np.finfo(float).eps * np.random.randn(*self.y_train_pred.shape),
            y_test_pred + np.finfo(float).eps * np.random.randn(*y_test_pred.shape),
            self.A_train, A_test)

        return pred
        