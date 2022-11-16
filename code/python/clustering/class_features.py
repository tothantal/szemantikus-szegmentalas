#!/usr/bin/python3
# -*- coding: utf-8 -*-

"""
Created on 2022. 09. 08. 18:49
@author  :  A. Tóth, Sz. Szeghalmy

A collect_class_features a kézi annotációk alapján minden osztályhoz összegyűjt bizonyos számú jellemzővektort.
A to_data_mat függvény az osztályozási eljárások által várt mátrixos alakra hozza a jellemzőket.


"""
import numpy as np
import pickle

import dataset_filter as df
import feature_extraction as fe

def collect_class_feature(d, img, depth, label, step = 8, limit = 20):
    """
    :param d: input-output dictionary
    :param img:
    :param depth:
    :param label:
    :param step: a kép bejárásánál használt lépésköz
    :param limit: maximum hány jellemzővektort gyűjtünk be egy osztályhoz egy képről
    :return:
    """

    feature_map = fe.create_feature_map(img, depth)

    # lassu, de csak 1x kell vegrehajtani
    for i in range(0, label.shape[0], step):
        for j in range(0, label.shape[1], step):
            class_lbl = label[i, j]
            if class_lbl not in d:
                d[class_lbl] = [feature_map[i, j, :]]
            elif len(d[class_lbl]) < limit:
                d[class_lbl].append(feature_map[i, j, :])
    

def collect_class_features(train_data):
    """

    :param train_data:
    :param step: a kép bejárásánál használt lépésköz
    :param limit: maximum hány jellemzővektort gyűjtünk be egy osztályhoz egy képről
    :return:
    """
    d = {}
    for img, depth, label in train_data:
        collect_class_feature(d, img, depth, label, 8, 20)
    return d

def to_data_mat(d):
    """

    :param d:
    :return:
    """

    labels = []
    samples = []
    for k, v in d.items():
        labels.append(np.repeat(k, len(v)))
        samples.append(np.vstack(v))


    return np.vstack(samples), np.hstack(labels)


# X = [[0], [1], [2], [3]]
# y = [0, 0, 1, 1]



if __name__ == "__main__":

    max_lim = 100  # csak a teszthez
    data = df.load_train_files('train.pickle')[ : max_lim]
    d = collect_class_features(data)
    X, y = to_data_mat(d)
    print(X.shape, y.shape)
    print(y)

    from sklearn.tree import DecisionTreeClassifier

    model = DecisionTreeClassifier()
    model.fit(X, y)

    pickle.dump(model, open("model.pickle", 'wb'))

    x1 = X[50, : ]
    print(model.predict([x1]))
