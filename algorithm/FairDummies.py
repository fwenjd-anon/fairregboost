import torch
import numpy as np
from algorithm.FairDummies_files.fair_dummies_learning import EquiClassLearner, EquiRegLearner

class FairDummies():
    """
    information [...]

    References:
        ...
    """
    def __init__(self,
                 df_dict,
                 learning_rate,
                 mu,
                 second_scale):
        """
        Args:
        """
        self.df_dict = df_dict
        self.lr = learning_rate
        self.mu = mu
        self.second_scale = second_scale


    def fit(self, X_train, y_train):
        """
        Information

        Args:

        Returns:
        """
        A_train = X_train[self.df_dict["sens_attrs"][0]]
        X_train = X_train.drop(self.df_dict["sens_attrs"], axis=1).values
        input_data_train = np.concatenate((A_train.values[:,np.newaxis], X_train), 1)

        self.model = EquiRegLearner(lr=self.lr, pretrain_pred_epochs=0, pretrain_dis_epochs=0, epochs=50,
            loss_steps=50, dis_steps=50, cost_pred=torch.nn.MSELoss(),
            in_shape=X_train.shape[1], batch_size=10000, model_type="deep_model", lambda_vec=self.mu,
            second_moment_scaling=self.second_scale, out_shape=1)

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
