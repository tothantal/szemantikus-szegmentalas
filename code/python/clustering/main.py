"""
Created on Tue Nov 15 15:47:38 2022
@author: Tóth Antal
"""

import dataset_filter as df
import feature_extraction as fe
import segmentation as se
import pickle
import numpy as np
import cv2

if __name__ == "__main__":
    test_files = df.load_test_files()
    features = {}

    model = pickle.load(open("model.pickle", 'rb'))

    for images in test_files:
        image = images[0]
        depth = images[1]

        preprocessed_image = fe.preprocess(image)
        segmented_image = se.create_segments_by_felzenszwalb(preprocessed_image)
        feature_map = fe.create_feature_map(image, depth)

        limit = 20
        step = 1
        features = {}
        for i in range(0, image.shape[0], step):
            for j in range(0, image.shape[1], step):
                segment_lbl = segmented_image[i, j]
                if segment_lbl not in features:
                    features[segment_lbl] = [feature_map[i, j, :]]
                elif len(features[segment_lbl]) < limit:
                    features[segment_lbl].append(feature_map[i, j, :])

        predicted = dict()
        for k in features.keys():
            mean_feature = np.array(features[k]).mean(axis = 0)

            predicted[k] = model.predict([mean_feature])

        dest = np.zeros(image.shape[:2], np.uint32)
        for i in range(0, image.shape[0]):
            for j in range(0, image.shape[1]):
                if segmented_image[i, j] in predicted:
                    dest[i, j] = predicted[segmented_image[i, j]]
        
                
        cv2.imshow("img", image)
        df.show_depth(depth)
        df.show_labels(dest, "result", 0)
