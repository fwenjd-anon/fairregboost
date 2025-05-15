"""
main method
"""
import warnings
import argparse
import ast
import copy
import itertools
import subprocess
import json
import shelve
import time
import joblib
import re
import random
import math
import traceback
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
import algorithm
from sklearn.linear_model import LogisticRegression, LinearRegression, SGDRegressor, Ridge, Lars, HuberRegressor, BayesianRidge, ElasticNet, Lasso
import xgboost as xgb

warnings.simplefilter(action='ignore', category=FutureWarning)
warnings.filterwarnings('ignore')

parser = argparse.ArgumentParser()
parser.add_argument("--ds", type=str, help="Name of the input .csv file.")
parser.add_argument("-o", "--output", type=str, help="Directory of the generated output files.")
parser.add_argument("--testsize", default=0.3, type=float, help="Dataset is randomly split into\
    training and test datasets. This value indicates the size of the test dataset. Default value: 0.5")
parser.add_argument("--index", default="index", type=str, help="Column name containing the index\
    of each entry. Default given column name: index.")
parser.add_argument("--sensitive", type=str, help="List of column names of the sensitive attributes.")
parser.add_argument("--label", type=str, help="Column name of the target value.")
parser.add_argument("--randomstate", default=-1, type=int, help="Randomstate of the splits.")
parser.add_argument("--models", default=None, type=str, help="List of models that should be trained.")
parser.add_argument("--tuning", default="False", type=str, help="Set to True if hyperparameter\
    tuning should be performed. Else, default parameter values are used. Default: False")
parser.add_argument("--baseline", default="False", type=str, help="")
args = parser.parse_args()

input_file = args.ds
link = args.output
testsize = float(args.testsize)
index = args.index
sens_attrs = ast.literal_eval(args.sensitive)
label = args.label
randomstate = args.randomstate
if randomstate == -1:
    import random
    randomstate = random.randint(1,1000)
model_list = ast.literal_eval(args.models)
tuning = args.tuning == "True"
baseline = args.baseline == "True"

df = pd.read_csv("Datasets/" + input_file + ".csv", index_col=index)
error_df = pd.read_csv("configs/ERRORS.csv")

X = df.loc[:, df.columns != label]
y = df[label].to_frame()

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3,
    random_state=randomstate)
X_train2 = copy.deepcopy(X_train)
X_test2 = copy.deepcopy(X_test)
y_train2 = copy.deepcopy(y_train)
y_test2 = copy.deepcopy(y_test)

result_df = copy.deepcopy(y_test)
for sens in sens_attrs:
    result_df[sens] = X_test[sens]
df_dict = dict()
df_dict["filename"] = input_file
df_dict["sens_attrs"] = sens_attrs
df_dict["label"] = label
df_dict["index"] = index

params = json.load(open('configs/params.json'))

failed_df = pd.DataFrame()

base_model_list = [
    ("LinearRegression", LinearRegression()),\
    ("Ridge", Ridge(alpha=1.0)),\
    ("BayesianRidge", BayesianRidge()),\
    ("XGBRegressor", xgb.XGBRegressor(
                                        n_estimators=100,  # Number of trees
                                        max_depth=3,       # Maximum depth of each tree
                                        learning_rate=0.1, # Learning rate
                                        subsample=0.8,     # Subsample of data
                                        colsample_bytree=0.8, # Subsample of features
                                        random_state=42
                                    )
    ),
]

if baseline:
    for base_model in base_model_list:
        clf = base_model[1]
        clf.fit(X_train, y_train)
        pred = clf.predict(X_test)
        result_df[base_model[0]] = pred
        result_df.to_csv(link + base_model[0] + "_prediction.csv", index_label="index")
        result_df = result_df.drop(columns=[base_model[0]])

for model in model_list:
    print(model)
    start = time.time()
    try:
        sol = False
        if tuning:
            paramlist = list(params[model]["tuning"].keys())
            parameters = []
            for param in paramlist:
                parameters.append(params[model]["tuning"][param])
            full_list = list(itertools.product(*parameters))
            do_eval = True
        else:
            paramlist = list(params[model]["default"].keys())
            li = []
            for param in paramlist:
                li.append(params[model]["default"][param])
            full_list = [li]
            do_eval = False

        max_val = -math.inf
        best_li = 0

        for i, li in enumerate(full_list):
            iteration_start = time.time()
            score = -math.inf
            try:    
                #pre-processing
                if model == "FairHeckman":
                    clf = algorithm.FairHeckman(df_dict, num_epochs=li[0], lr=li[1])
                #in-processing
                elif model == "AdversarialDebiasing":
                    clf = algorithm.AdversarialDebiasing(df_dict, learning_rate=li[0], mu=li[1], epochs=li[2])
                elif model == "FairDummies":
                    clf = algorithm.FairDummies(df_dict, learning_rate=li[0], mu=li[1], second_scale=li[2])
                elif model == "HGR":
                    clf = algorithm.HGR(df_dict, learning_rate=li[0], mu=li[1])
                elif model == "FairGeneralizedLinearModel":
                    clf = algorithm.FairGeneralizedLinearModelClass(df_dict, lam=li[0], discretization=li[1])
                elif model == "ConvexFrameworkModel":
                    clf = algorithm.ConvexFrameworkModelClass(df_dict, lam=li[0], penalty=li[1])
                elif model == "HSICLinearRegression":
                    clf = algorithm.HSICLinearRegressionClass(df_dict, lam=li[0])
                elif model == "GeneralFairERM":
                    clf = algorithm.GeneralFairERMClass(df_dict, eps=li[0])
                elif model == "ReductionsApproach":
                    clf = algorithm.ReductionsApproachClass(df_dict, c=li[0])
                #post-processing
                elif model == "FairWassBary":
                    clf = algorithm.FairWassBary(df_dict)
                elif model == "Wass2Geo":
                    clf = algorithm.Wass2Geo(df_dict, t=li[0], bins=li[1])
                elif model == "FairPlugRecal":
                    clf = algorithm.FairPlugRecal(df_dict, beta=li[0])
                elif model == "PrivateHDEFairPostProcessor":
                    clf = algorithm.PrivateHDEFairPostProcessor(df_dict, alpha=li[0], bins=li[1], eps=li[2])
                elif model == "UnawareFairReg":
                    clf = algorithm.UnawareFairReg(df_dict, base=li[0], L=li[1], eps=li[2])
                elif model == "FairRegBoost":
                    clf = algorithm.FairRegBoost(df_dict, preparation_nr=li[0], uc_strategy=li[1], lam=li[2], gamma=li[3])

                X_train = copy.deepcopy(X_train2)
                X_test = copy.deepcopy(X_test2)
                y_train = copy.deepcopy(y_train2)
                y_test = copy.deepcopy(y_test2)

                clf.fit(X_train, y_train)
                pred = clf.predict(X_test)

                if tuning and pred is not None:
                    result_df[model + "_" + str(i)] = pred
                    result_df.to_csv(link + model + "_" + str(i) + "_prediction.csv", index_label="index")
                    result_df = result_df.drop(columns=[model + "_" + str(i)])
                elif pred is not None:
                    result_df[model] = pred
                    result_df.to_csv(link + model + "_prediction.csv", index_label="index")
                    result_df = result_df.drop(columns=[model])


            except Exception as e:
                print("------------------")
                pred = None
                failcount = len(failed_df)
                failed_df.at[failcount, "model"] = model
                failed_df.at[failcount, "exceptions"] = e
                print(model)
                print(e)
                print(traceback.format_exc())
                print("------------------")

    except Exception as E:
        print(E)
        err_count = len(error_df)
        error_df.at[err_count, "dataset"] = input_file
        error_df.at[err_count, "model"] = model
        error_df.at[err_count, "error_type"] = str(type(E))
        error_df.at[err_count, "error_msg"] = str(E)

error_df.to_csv("configs/ERRORS.csv", index=False)
