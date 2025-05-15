import torch
import numpy as np
from algorithm.FairDummies_files.others.continuous_fairness import HGR_Class_Learner, HGR_Reg_Learner

class HGR():
    """
    information [...]

    References:
        ...
    """
    def __init__(self,
                 df_dict,
                 learning_rate,
                 mu):
        """
        Args:
        """
        self.df_dict = df_dict
        self.lr = learning_rate
        self.mu = mu


    def fit(self, X_train, y_train):
        """
        Information

        Args:

        Returns:
        """
        size_check = (len(X_train)-1)/32
        if size_check.is_integer():
            X_train = X_train[:-1]
            y_train = y_train[:-1]
        A_train = X_train[self.df_dict["sens_attrs"][0]]
        X_train = X_train.drop(self.df_dict["sens_attrs"], axis=1).values
        input_data_train = np.concatenate((A_train.values[:,np.newaxis], X_train), 1)

        self.model = HGR_Reg_Learner(lr=self.lr, epochs=50, mu=self.mu, cost_pred=torch.nn.MSELoss(),
            in_shape=X_train.shape[1], out_shape=1, batch_size=256, model_type="deep_model")

        self.model.fit(input_data_train, y_train.to_numpy().reshape((y_train.to_numpy().shape[0],)))

        return self


    def predict(self, X_test):
        """
        Information

        Args:

        Returns:
        """
        A_test = X_test[self.df_dict["sens_attrs"][0]]
        X_test = X_test.drop(self.df_dict["sens_attrs"], axis=1).values
        input_data_test = np.concatenate((A_test.values[:,np.newaxis], X_test), 1)

        pred = self.model.predict(input_data_test)

        return pred
