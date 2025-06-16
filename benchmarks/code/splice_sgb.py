"""
Author: Haoyin Xu
"""

import time
import psutil
import argparse
from numpy.random import permutation
import openml
from sklearn.model_selection import train_test_split

# from sklearn.tree import DecisionTreeClassifier
# from sklearn.ensemble import RandomForestClassifier
# from river import tree
# from skgarden import MondrianForestClassifier
# from sdtf import StreamDecisionForest
from capymoa.classifier import StreamingGradientBoostedTrees
from capymoa.stream import NumpyStream

from toolbox import *

import warnings

warnings.filterwarnings("ignore")


def experiment_sgb():
    """Runs experiments for SGBT"""
    sgb_l = []
    train_time_l = []
    test_time_l = []
    v_m_l = []
    n_node_l = []
    size_l = []

    stream = NumpyStream(X_r, y_r, dataset_name="splice")
    schema = stream.get_schema()

    sgb = StreamingGradientBoostedTrees(
        schema,
        boosting_iterations=10,
        percentage_of_features=int(np.sqrt(X_r.shape[1]) / X_r.shape[1] * 100),
    )
    p = psutil.Process()

    for i in range(2300):
        instance = stream.next_instance()

        start_time = time.perf_counter()
        sgb.train(instance)
        end_time = time.perf_counter()
        train_time_l.append(end_time - start_time)

        if i > 0 and (i + 1) % 100 == 0:
            # Check size
            # size = clf_size(sgb, DIRECTORY + "results/temp.pickle")
            # size_l.append(size)

            # Check memory
            v_m = (
                p.memory_full_info().rss / 1024 / 1024 / 1024,
                p.memory_full_info().vms / 1024 / 1024 / 1024,
            )
            v_m_l.append(v_m)

            # Check node counts
            # n_node = sgb.n_nodes
            # n_node_l.append(n_node)

            p_t = 0.0
            test_stream = NumpyStream(X_test, y_test, dataset_name="splice")
            start_time = time.perf_counter()
            for j in range(X_test.shape[0]):
                test_instance = test_stream.next_instance()
                y_pred = sgb.predict(test_instance)
                if y_pred == y_test[j]:
                    p_t += 1
            sgb_l.append(p_t / X_test.shape[0])
            end_time = time.perf_counter()
            test_time_l.append(end_time - start_time)

    # Reformat the train times
    new_train_time_l = []
    for i in range(1, 2300):
        train_time_l[i] += train_time_l[i - 1]
        if i > 0 and (i + 1) % 100 == 0:
            new_train_time_l.append(train_time_l[i])
    train_time_l = new_train_time_l

    return sgb_l, train_time_l, test_time_l, v_m_l, n_node_l, size_l


DIRECTORY = "/mnt/ssd1/hao/sdtf/"

# Prepare splice DNA data
dataset = openml.datasets.get_dataset(46)
X, y, is_categorical, _ = dataset.get_data(
    dataset_format="array", target=dataset.default_target_attribute
)
X_train, X_test, y_train, y_test = train_test_split(X, y)

# Parse classifier choices
parser = argparse.ArgumentParser()
parser.add_argument("-all", help="all classifiers", required=False, action="store_true")
parser.add_argument("-sgb", help="sgbt", required=False, action="store_true")
args = parser.parse_args()

# Perform experiments

if args.all or args.sgb:
    sgb_acc_l = []
    sgb_train_t_l = []
    sgb_test_t_l = []
    sgb_v_m_l = []
    # sgb_n_node_l = []
    sgb_size_l = []
    for i in range(1):
        p = permutation(X_train.shape[0])

        X_r = X_train[p]
        y_r = y_train[p]

        sgb_acc, sgb_train_t, sgb_test_t, sgb_v_m, sgb_n_node, sgb_size = (
            experiment_sgb()
        )
        sgb_acc_l.append(sgb_acc)
        sgb_train_t_l.append(sgb_train_t)
        sgb_test_t_l.append(sgb_test_t)
        sgb_v_m_l.append(sgb_v_m)
        # sgb_n_node_l.append(sgb_n_node)
        # sgb_size_l.append(sgb_size)

        write_result(DIRECTORY + "results/sgb/splice_acc", sgb_acc_l)
        write_result(DIRECTORY + "results/sgb/splice_train_t", sgb_train_t_l)
        write_result(DIRECTORY + "results/sgb/splice_test_t", sgb_test_t_l)
        write_result(DIRECTORY + "results/sgb/splice_v_m", sgb_v_m_l, True)
        # write_result(DIRECTORY + "results/sgb/splice_n_node", sgb_n_node_l)
        # write_result(DIRECTORY + "results/sgb/splice_size", sgb_size_l, True)
