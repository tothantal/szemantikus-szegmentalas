# -*- coding: utf-8 -*-
"""
Created on Tue Nov 15 15:47:38 2022

@author: Tóth Antal
"""

import dataset_filter as df
import feature_extraction as fe
import segmentation as se
import class_features as cf
import pickle
import numpy as np

if __name__ == "__main__":
    test_files = df.load_test_files()
    features = {}
    
    for images in test_files:
        image = images[0]
        depth = images[1]
        label = images[2]
    
    segmented_image = se.kmeans(image, 2)
    preprocessed_image = fe.preprocess(segmented_image)
    feature_map = fe.create_feature_map(preprocessed_image, depth)
    
    limit = 20
    step = 1
    features = {}
    for i in range(0, label.shape[0], step):
        for j in range(0, label.shape[1], step):
            class_lbl = label[i, j]
            if class_lbl not in features:
                features[class_lbl] = [feature_map[i, j, :]]
            elif len(features[class_lbl]) < limit:
                features[class_lbl].append(feature_map[i, j, :])
    
    X, _ = cf.to_data_mat(features)
    
    model = pickle.load(open("model.pickle", 'rb'))
    x1 = X[50, : ]
    print(model.predict([x1]))
    print(model.kneighbors([x1], return_distance=True))
    
