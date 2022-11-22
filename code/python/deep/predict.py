# -*- coding: utf-8 -*-
"""
Created on Mon Nov 21 20:39:38 2022

@author: Tóth Antal
"""

import pickle
import tensorflow as tf
import deep as d
import numpy as np
import cv2


def load_test_files(fname = 'test.pickle'):
    """ list of (rgb [:,:,3], depth [:,:], label [:,:]) image tuples """
    return pickle.load(open(fname, 'rb'))

def get_model():
    model = tf.keras.models.load_model('deep.model', 
                                       custom_objects = {
                                           'f1_metric':d.f1_metric
                                           }  
                                       )
    return model

def show_labels(img, title = "Labels", wait = 0):
    img2 = img.astype(np.uint8)
    cv2.imshow(title, img2)
    cv2.waitKey(wait)

if __name__ == "__main__":
    test_files = load_test_files()
    model = get_model()
    
    imgs = []
    for images in test_files:
        image = images[0]
        imgs.append(image)
        
        
    pred = model.predict(np.array(imgs))
    
    i = 0
    for p in pred:
        image = np.array(p)
        image = np.moveaxis(image, -1, 0)
        for img in image:
            # show_labels(i, "result", 0)
            cv2.imwrite("results/result" + str(i) + ".png", img.astype(np.uint8))
            i = i + 1
