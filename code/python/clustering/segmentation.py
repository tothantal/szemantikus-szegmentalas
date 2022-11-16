# -*- coding: utf-8 -*-
"""
Created on Tue Nov 15 15:50:05 2022

@author: Tóth Antal
"""

import dataset_filter as df
import cv2
import numpy as np
import skimage.io
import skimage.segmentation

def create_segments_by_kmeans(img, k = 50):
    h, w = img.shape[0:2]
    data = img.reshape((h * w, img.shape[2]))
    data = np.float32(data)

    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 3, 0.002)
    compactness, labels, centers = cv2.kmeans(data, k, None, criteria, 3, cv2.KMEANS_RANDOM_CENTERS )
    label_img = np.uint32(labels.reshape((h, w)))
    return label_img

def create_segments_by_felzenszwalb(image):
    segments = skimage.segmentation.felzenszwalb(image, scale=150, sigma=1.0, min_size=25)
    return segments


if __name__ == "__main__":
    train_files = df.load_train_files()
    
    for images in train_files:
        image = images[0]
        
    kmeans_segmented = create_segments_by_kmeans(image)
    fh_segmented = create_segments_by_felzenszwalb(image)
    
