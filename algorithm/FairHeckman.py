import numpy as np
import pandas as pd
import torch
from torch import nn
from sklearn.metrics import mean_squared_error
from scipy.stats import norm
from numpy.linalg import inv
from algorithm.FairSampling_files.heckman import linear_regression_heckman_correction_fairness
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

class linearRegression(nn.Module):
    def __init__(self, size):
        super(linearRegression, self).__init__()
        self.linear = nn.Linear(size, 1)  # input and output is 1 dimension

    def forward(self, x_total, x_black, x_non_black):
        y_pred = self.linear(x_total)
        y_pred_black = self.linear(x_black)
        y_pred_non_black = self.linear(x_non_black)
        return y_pred, y_pred_black, y_pred_non_black

class FairHeckman:
    def __init__(self, df_dict, num_epochs=100, lr=1e-1):
        self.df_dict = df_dict
        self.label = df_dict["label"]
        self.sensitive = df_dict["sens_attrs"][0]
        self.num_epochs = num_epochs
        self.lr = lr
        self.model = None
        torch.manual_seed(0)


    def fit(self, X_train, y_train):

        X_observe, X_unobserve, y_observe, y_unobserve = train_test_split(X_train, y_train, test_size=0.3, random_state=42, shuffle=False)

        heck_y_train = y_observe.copy()
        heck_y_test = y_unobserve.copy()
        heck_y_test[heck_y_test < 2.0] = np.nan

        self.sc = StandardScaler()
        X_observe = self.sc.fit_transform(X_observe)
        X_unobserve = self.sc.transform(X_unobserve)
        X_total = np.r_[X_observe, X_unobserve]
        y_total = np.r_[y_observe, y_unobserve]

        X_selection = pd.DataFrame(X_total)
        X_prediction = pd.DataFrame(X_total)
        X_observe_selection = pd.DataFrame(X_observe)
        X_observe_prediction = pd.DataFrame(X_observe)

        heckman_y = np.r_[heck_y_train, heck_y_test].reshape(-1,)
        A_train = X_train[self.sensitive].to_numpy().reshape(-1,)

        y_observe = y_observe.to_numpy().reshape(-1,)
        y_unobserve = y_unobserve.to_numpy().reshape(-1,)

        X_train, X_train_discriminated, X_train_non_discriminated, y_train, y_train_discriminated, y_train_non_discriminated, self.params_selection = linear_regression_heckman_correction_fairness(
            X_selection, X_prediction, heckman_y, X_train, y_observe, y_unobserve, y_train, A_train)


        X_train, X_train_discriminated, X_train_non_discriminated = torch.from_numpy(X_train), torch.from_numpy(X_train_discriminated), torch.from_numpy(X_train_non_discriminated)
        X_train, X_train_discriminated, X_train_non_discriminated = X_train.float(), X_train_discriminated.float(), X_train_non_discriminated.float()

        y_train, y_train_discriminated, y_train_non_discriminated = np.asarray(y_train), np.asarray(y_train_discriminated), np.asarray(y_train_non_discriminated)
        y_train, y_train_discriminated, y_train_non_discriminated = torch.from_numpy(y_train), torch.from_numpy(y_train_discriminated), torch.from_numpy(y_train_non_discriminated)
        y_train, y_train_discriminated, y_train_non_discriminated = y_train.float(), y_train_discriminated.float(), y_train_non_discriminated.float()
        
        
        self.model = linearRegression(X_train.shape[1])
        criterion = nn.MSELoss()
        optimizer = torch.optim.SGD(self.model.parameters(), lr=self.lr)

        for epoch in range(self.num_epochs):
            x_total, x_discriminated, x_non_discriminated = X_train, X_train_discriminated, X_train_non_discriminated
            y, y_discriminated, y_non_discriminated = y_train, y_train_discriminated, y_train_non_discriminated
            # forward
            y_pred, y_pred_discriminated, y_pred_non_discriminated = self.model(x_total, x_discriminated, x_non_discriminated)
            y_train_pred, y_train_discriminated_pred, y_train_non_discriminated_pred = y_pred, y_pred_discriminated, y_pred_non_discriminated
            acc_loss = criterion(y, y_pred)
            fair_loss = criterion(y_discriminated, y_pred_discriminated) - criterion(y_non_discriminated, y_pred_non_discriminated)
            # backward
            loss = acc_loss + 0.5*abs(fair_loss)*abs(fair_loss)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            #if (epoch+1) % 50 == 0:
             #   print(f'Epoch[{epoch+1}/{self.num_epochs}], loss: {loss.item():.6f}')

    
    def predict(self, X_test):
        X_test2 = self.sc.transform(X_test)
        A_test = X_test[self.sensitive].to_numpy().reshape(-1,)

        lambda_u = X_test2.dot(self.params_selection)
        inverse_mills = norm.pdf(lambda_u) / norm.cdf(lambda_u)

        inverse_mills = inverse_mills.reshape(len(inverse_mills), 1)
        X_test2 = np.hstack((X_test2, inverse_mills))

        X_test_discriminated = np.array([X_test2[i] for i in range(len(A_test)) if A_test[i] == 1.0])
        X_test_non_discriminated = np.array([X_test2[i] for i in range(len(A_test)) if A_test[i] == 0.0])

        X_test, X_test_discriminated, X_test_non_discriminated = torch.from_numpy(X_test2), torch.from_numpy(X_test_discriminated), torch.from_numpy(X_test_non_discriminated)
        X_test, X_test_discriminated, X_test_non_discriminated = X_test.float(), X_test_discriminated.float(), X_test_non_discriminated.float()
        
        with torch.no_grad():
            y_pred, y_pred_discriminated, y_pred_non_discriminated = self.model(X_test, X_test_discriminated, X_test_non_discriminated)


        return y_pred