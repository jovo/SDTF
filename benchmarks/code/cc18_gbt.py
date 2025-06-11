"""
Author: Haoyin Xu
"""

import time
import argparse
from numpy.random import permutation
import json
import openml
# import sklearn
from sklearn.model_selection import StratifiedKFold, train_test_split
# from sklearn.ensemble import GradientBoostingClassifier
from xgboost import XGBClassifier
import warnings

warnings.filterwarnings("ignore")

from toolbox import *


def experiment_xgb(X_train, X_test, y_train, y_test):
    """Runs experiments for Random Forest"""
    xgb_l = []
    train_time_l = []
    test_time_l = []

    xgb = XGBClassifier(n_estimators=100, random_state=23)
    batch_counts = len(y_train) / BATCH_SIZE
    for i in range(int(batch_counts)):
        train_size = (i + 1) * BATCH_SIZE
        if train_size>=len(y_train)-len(np.unique(y_train)):
            X_t = X_train
            y_t = y_train
        else:
            X_t, _, y_t, _  = train_test_split(X_train, y_train, train_size=train_size, stratify=y_train)
        # X_t = X_train[: (i + 1) * BATCH_SIZE]
        # y_t = y_train[: (i + 1) * BATCH_SIZE]

        # Train the model
        start_time = time.perf_counter()
        xgb.fit(X_t, y_t)
        end_time = time.perf_counter()
        train_time_l.append(end_time - start_time)

        # Test the model
        start_time = time.perf_counter()
        xgb_l.append(prediction(xgb, X_test, y_test))
        end_time = time.perf_counter()
        test_time_l.append(end_time - start_time)

    return xgb_l, train_time_l, test_time_l


# Parse classifier choices
parser = argparse.ArgumentParser()
parser.add_argument("-all", help="all classifiers", required=False, action="store_true")
parser.add_argument("-xgb", help="gradient boost", required=False, action="store_true")
args = parser.parse_args()

BATCH_SIZE = 100

xgb_acc_dict = {}
xgb_train_t_dict = {}
xgb_test_t_dict = {}

DATA_IDS = [
    # 3,
    # 6,
    # 11,
    # 12,
    # 14,
    # 15,
    # 16,
    # 18,
    # 22,
    # 23,
    # 28,
    # 29,
    # 31,
    # 32,
    # 37,
    # 44,
    # 46,
    # 50,
    # 54,
    # 151,
    # 182,
    # 188,
    # 38,
    # 307,
    # 300,
    # 458,
    # 469,
    # 554,
    # 1049,
    # 1050,
    # 1053,
    # 1063,
    # 1067,
    # 1068,
    # 1590,
    4134,
    1510,
    1489,
    1494,
    1497,
    1501,
    1480,
    1485,
    1486,
    1487,
    1468,
    1475,
    1462,
    1464,
    4534,
    6332,
    1461,
    4538,
    1478,
    23381,
    40499,
    40668,
    40966,
    40982,
    40994,
    40983,
    40975,
    40984,
    40979,
    40996,
    41027,
    23517,
    40923,
    40927,
    40978,
    40670,
    40701,
]

# Prepare cc18 data
# for data_id in openml.study.get_suite("OpenML-CC18").data:
for data_id in DATA_IDS:
    # Retrieve dataset
    dataset = openml.datasets.get_dataset(data_id)
    X, y, is_categorical, _ = dataset.get_data(
        dataset_format="array", target=dataset.default_target_attribute
    )
    # X, y = sklearn.datasets.fetch_openml(data_id=data_id, return_X_y=True, data_home="/mnt/ssd1/hao/stream/data/")
    X = np.nan_to_num(X)
    # y = y.to_numpy()

    # enc = sklearn.preprocessing.OneHotEncoder(handle_unknown='ignore')
    # X = enc.fit_transform(X, y)

    xgb_acc_dict[data_id] = []
    xgb_train_t_dict[data_id] = []
    xgb_test_t_dict[data_id] = []

    # Split the datasets into 5-fold CV
    skf = StratifiedKFold(shuffle=True)
    for train_index, test_index in skf.split(X, y):
        print(data_id, 1)
        X_train, X_test, y_train, y_test = (
            X[train_index],
            X[test_index],
            y[train_index],
            y[test_index],
        )

        # Shuffle the training sets
        p = permutation(len(X_train))
        X_train = X_train[p]
        y_train = y_train[p]

        xgb_acc, xgb_train_t, xgb_test_t = experiment_xgb(X_train, X_test, y_train, y_test)
        xgb_acc_dict[data_id].append(xgb_acc)
        xgb_train_t_dict[data_id].append(xgb_train_t)
        xgb_test_t_dict[data_id].append(xgb_test_t)

        f = open("/mnt/ssd1/hao/stream/xgb_cc18_acc_4134.json", "w")
        json.dump(xgb_acc_dict, f)
        f.close()

        f = open("/mnt/ssd1/hao/stream/xgb_cc18_train_t_4134.json", "w")
        json.dump(xgb_train_t_dict, f)
        f.close()

        f = open("/mnt/ssd1/hao/stream/xgb_cc18_test_t_4134.json", "w")
        json.dump(xgb_test_t_dict, f)
        f.close()
