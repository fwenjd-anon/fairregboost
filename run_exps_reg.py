"""
Python file/script for evaluation purposes.
"""
import subprocess
import os
import copy
import random
import pandas as pd

DATA_DICT = {
    "lsac_race": {"sens_attrs": ["race"], "label": "target"},
    "lsac_gender": {"sens_attrs": ["gender"], "label": "target"},
    "lsac_multi": {"sens_attrs": ["race", "gender"], "label": "target"},
    "parkinsons_gender": {"sens_attrs": ["gender"], "label": "target"},
    "parkinsons_age": {"sens_attrs": ["age"], "label": "target"},
    "parkinsons_multi": {"sens_attrs": ["gender", "age"], "label": "target"},
    "student_performance": {"sens_attrs": ["sex"], "label": "target"},
    "communities_foreigners": {"sens_attrs": ["Foreigners"], "label": "ViolentCrimesPerPop"},
    "communities_race": {"sens_attrs": ["race"], "label": "ViolentCrimesPerPop"},
    "communities_multi": {"sens_attrs": ["race", "Foreigners"], "label": "ViolentCrimesPerPop"},
    "ACS_ID": {"sens_attrs": ["SEX"], "label": "PINCP"},
    "ACS_AK": {"sens_attrs": ["SEX"], "label": "PINCP"},
    "ACS_TN": {"sens_attrs": ["SEX"], "label": "PINCP"},
    }

"""
PARAMETER SETTINGS START
"""
models = [
          "ReductionsApproach", "FairHeckman",
          "FairGeneralizedLinearModel", "GeneralFairERM",
          "ConvexFrameworkModel", "HSICLinearRegression",
          "FairDummies", "HGR", "AdversarialDebiasing",
          "Wass2Geo", "FairWassBary", "UnawareFairReg",
          "PrivateHDEFairPostProcessor", "FairRegBoost"
          ]

tuning = True

ds = "communities_foreigners"
sensitive = DATA_DICT[ds]["sens_attrs"]
label = DATA_DICT[ds]["label"]

#If -1 a random one will be selected
randomstate = -1
    
link = "Results/" + str(ds) + "/"

try:
    os.makedirs(link)
except FileExistsError:
    pass

try:
    subprocess.check_call(['python', '-Wignore', 'main.py', '--output', str(link),
        '--ds', str(ds), '--sensitive', str(sensitive), '--label', str(label),
        '--models', str(models), '--tuning', str(tuning), '--randomstate', str(randomstate)])
except Exception as e:
    print(e)


#For evaluation add models to evaluate
model_list_eval = []
for model in models:
    for i in range(200):
        model_list_eval.append(model + "_" + str(i))

try:
    subprocess.check_call(['python', '-Wignore', 'evaluation_reg.py', '--folder', str(link),
        '--ds', str(ds), '--sensitive', str(sensitive), '--label', str(label),
        '--models', str(model_list_eval), '--name', 'EVALUATION'])
except Exception as e:
    print(e)

