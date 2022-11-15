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

def kmeans(img, clusters):
    rgb_image = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    
    pixel_values = rgb_image.reshape((-1, 3))
    pixel_values = np.float32(pixel_values)
    
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 100, 0.2)
    
    k = clusters 
    
    _, labels, (centers) = cv2.kmeans(pixel_values, k, None, criteria, 21, cv2.KMEANS_RANDOM_CENTERS)
    
    centers = np.uint8(centers)
    labels = labels.flatten()
    
    segmented_image = centers[labels.flatten()]
    segmented_image = segmented_image.reshape(rgb_image.shape)
    
    return segmented_image
    
def felzenszwalb_huttenlocher(img, show = False):
    segments = skimage.segmentation.felzenszwalb(img, scale = 100, sigma = 0.5, min_size = 25)
    if show == True:
        cv2.imshow("Segmented Image", skimage.segmentation.mark_boundaries(img, segments, color = (0, 0, 1.0)))
    return segments


if __name__ == "__main__":
    train_files = df.load_train_files()
    
    for images in train_files:
        image = images[0]
        
    kmeans_segmented = kmeans(image, 7)
    felzenszwalb_huttenlocher(image, show = True)
    
