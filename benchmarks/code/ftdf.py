# import libraries
import openml
import tarfile
import os
import numpy as np
import time
import pandas as pd
import warnings
import string

warnings.filterwarnings("ignore")
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier

from sklearn.model_selection import StratifiedKFold

from sklearn.model_selection import train_test_split
from numpy.random import default_rng

from treeple.experimental import StreamDecisionForest
from treeple.tree import HonestTreeClassifier
from treeple.ensemble import HonestForestClassifier

import pickle

from sklearn.metrics import (
    balanced_accuracy_score,
    mean_absolute_error,
    mean_squared_error,
    roc_auc_score,
    roc_curve,
    accuracy_score,
)
from copy import deepcopy


# In[2]:

if __name__ == "__main__":
    # Load the real datasets
    DIRECTORY = "/mnt/ssd1/hao/ftdf/"
    SEED = 23
    N_JOBS = 53
    # IDX = 41
    # IDX_END = 42
    N_ITR = 10
    N_EST = 100
    LEAF_SIZE = 500
    # SIZE_L = [0, 10, 15, 18, 20, 22, 26, 30, 40, 50, 65, 70, 75, 80, 100]
    # DEPTH_L = [1, 2, 3, 4]

    # data_ids = [554]

    dataset = openml.datasets.get_dataset(554)
    X, y, is_categorical, _ = dataset.get_data(
        dataset_format="dataframe", target=dataset.default_target_attribute
    )

    FEATURENAME = "mnist"

    dataset_2 = openml.datasets.get_dataset(40996)
    X_2, y_2, is_categorical, _ = dataset_2.get_data(
        dataset_format="dataframe", target=dataset.default_target_attribute
    )

    for j in range(N_ITR):
        est_l = []
        pos_l = []
        y_l = []
        train_time_l = []
        acc_l = []
        node_l = []

        kf = StratifiedKFold(n_splits=5, shuffle=True, random_state=j)
        for train_index, test_index in kf.split(X, y):
            print(j, 1)
            X_train, X_test, y_train, y_test = (
                X.iloc[train_index],
                X.iloc[test_index],
                y.iloc[train_index],
                y.iloc[test_index],
            )
            # y_l.append(y_test.astype(int))

            # Shuffle the training sets
            p = default_rng(seed=SEED).permutation(len(X_train))
            X_t = X_train.iloc[p]
            y_t = y_train.iloc[p].astype(int)

            # prepare base tree trained on MNIST
            model = HonestForestClassifier(
                n_estimators=N_EST,
                honest_fraction=0.00001,
                random_state=j,
                honest_prior="empirical",
                honest_method="apply",
                n_jobs=N_JOBS,
                kernel_method=False,
                stratify=False,
            )

            # Train the model
            start_time = time.perf_counter()
            model.fit(X_t, y_t)
            end_time = time.perf_counter()
            train_time_l.append(end_time - start_time)

            # node count of original forest
            # origin_count = model.tree_.node_count
            # node_l.append(origin_count)

            # reserve the original copy
            model_copy = deepcopy(model)
            est_l.append(model_copy)

            X_train_2, X_test_2, y_train_2, y_test_2 = (
                X_2.iloc[train_index],
                X_2.iloc[test_index],
                y_2.iloc[train_index],
                y_2.iloc[test_index].astype(int),
            )

            y_l.append(y_test_2)

            # Shuffle the training sets
            X_t_2 = X_train_2.iloc[p]
            y_t_2 = y_train_2.iloc[p].astype(int)

            # for size in SIZE_L:
            model = deepcopy(model_copy)

            for tree in model.estimators_:
                tree.honest_indices_ = np.arange(LEAF_SIZE)

                tree._fit_leaves(
                    X_t_2[:LEAF_SIZE],
                    y_t_2[:LEAF_SIZE],
                    sample_weight=np.ones(LEAF_SIZE),
                )

            est_l.append(model)
            pos_l.append(model.predict_proba(X_test_2))
            acc_l.append(
                accuracy_score(
                    y_test_2, np.argmax(model.predict_proba(X_test_2), axis=1)
                )
            )

            # for dep in DEPTH_L:
            model = RandomForestClassifier(
                n_estimators=N_EST,
                random_state=j,
                n_jobs=N_JOBS,
            )

            start_time = time.perf_counter()
            model.fit(X_t_2[:LEAF_SIZE], y_t_2[:LEAF_SIZE])
            end_time = time.perf_counter()
            train_time_l.append(end_time - start_time)

            # node_l.append(model.tree_.node_count)

            est_l.append(model)
            pos_l.append(model.predict_proba(X_test_2))
            acc_l.append(
                accuracy_score(
                    y_test_2, np.argmax(model.predict_proba(X_test_2), axis=1)
                )
            )

            with open(
                DIRECTORY
                + "results/ftdf_100_est_"
                + FEATURENAME
                + "_"
                + str(j)
                + ".pkl",
                "wb",
            ) as f:
                pickle.dump(est_l, f)

            with open(
                DIRECTORY
                + "results/ftdf_100_pos_"
                + FEATURENAME
                + "_"
                + str(j)
                + ".pkl",
                "wb",
            ) as f:
                pickle.dump(pos_l, f)

            with open(
                DIRECTORY
                + "results/ftdf_100_train_"
                + FEATURENAME
                + "_"
                + str(j)
                + ".pkl",
                "wb",
            ) as f:
                pickle.dump(train_time_l, f)

            with open(
                DIRECTORY + "results/ftdf_100_y_" + FEATURENAME + "_" + str(j) + ".pkl",
                "wb",
            ) as f:
                pickle.dump(y_l, f)

            # with open(
            #     DIRECTORY + "results/ftdf_100_node_" + FEATURENAME + "_" + str(j) + ".pkl",
            #     "wb",
            # ) as f:
            #     pickle.dump(node_l, f)

            with open(
                DIRECTORY
                + "results/ftdf_100_acc_"
                + FEATURENAME
                + "_"
                + str(j)
                + ".pkl",
                "wb",
            ) as f:
                pickle.dump(acc_l, f)
