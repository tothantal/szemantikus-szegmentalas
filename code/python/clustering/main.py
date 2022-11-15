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

if __name__ == "__main__":
    test_files = df.load_test_files()
    
    for images in test_files:
        image = images[0]
        depth = images[1]
    
    segmented_image = se.kmeans(image, 7)
    preprocessed_image = fe.preprocess(segmented_image)
    feature_map = fe.create_feature_map(preprocessed_image, depth)
    
    model = pickle.load(open("model.pickle", 'rb'))
