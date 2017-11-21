# code to prepare samples for training
import numpy as np

import pickle


__all__ = ["read_data", "split_data", "random_partition",
            "select_data"]

# read from "csv file"
def read_data(file, cast):
    f = open(file)
    l = list()
    for line in f:
        l.append(line.replace("\n", "").replace('"', "").split(" "))
    features = l[0]
    data = np.array(list(map(cast, l[1:])))
    return (features, data)



def random_partition(data, testing_size, total_size, ind=None, o_file=None):
    if ind == None:
        ind = set(np.random.random_integers(0, total_size - 1, testing_size))
        while len(ind) < testing_size:
            candidate = np.random.randint(0, total_size)
            if candidate not in ind:
                ind.add(candidate)
    if o_file != None:
        with open(o_file, "wb") as ostream:
            pickle.dump(ind, ostream)

    features = len(data[0])
    training_size = total_size - testing_size
    training = np.zeros((training_size, features))
    test = np.zeros((testing_size, features))

    t_c = 0 # testing sample counter
    r_c = 0 # training sample counter

    for i in range(0, total_size):
        if i in ind:
            for j in range(features):
                test[t_c][j] = data[i][j]
            t_c += 1
        else:
            for j in range(features):
                training[r_c][j] = data[i][j]
            r_c = r_c + 1
    return (training, test)



def select_data(data, feature_columns, goal_columns=None):
    features = len(feature_columns)
    f_map = dict()
    for i in range(features):
        f_map[feature_columns[i]] = i

    r_size = len(data[0])
    t_size = len(data[1])

    training = np.zeros((r_size, features))
    test = np.zeros((t_size, features))

    for i in range(r_size):
        for j in feature_columns:
            training[i][f_map[j]] = data[0][i][j]

    for i in range(t_size):
        for j in feature_columns:
            test[i][f_map[j]] = data[1][i][j]

    return (training, test)


# simple data splitting collecting indexes
def split_data(data, testing_size, total_size, feature_columns, goal_columns):
    ind = set(np.random.random_integers(0, total_size - 1, testing_size))
    while len(ind) < testing_size:
        candidate = np.random.randint(0, total_size)
        if candidate not in ind:
            ind.add(candidate)

    features = len(feature_columns)
    f_map = dict()
    for i in range(features):
        f_map[feature_columns[i]] = i

    training_size = total_size - testing_size
    training = [
        np.zeros((training_size, features)),
        np.zeros(training_size)]
    test = [
        np.zeros((testing_size, features)), 
        np.zeros(testing_size)]

    t_c = 0 # testing sample counter
    r_c = 0 # training sample counter

    for i in range(0, total_size):
        if i in ind:
            #for j in goal_columns:
            test[1][t_c] = data[i][goal_columns]
            for j in feature_columns:
                test[0][t_c][f_map[j]] = data[i][j]
            t_c = t_c + 1
        else:
            #for j in goal_columns:
            training[1][r_c] = data[i][goal_columns]
            for j in feature_columns:
                training[0][r_c][f_map[j]] = data[i][j]
            r_c = r_c + 1

    return (training[0], training[1], test[0], test[1], ind)
