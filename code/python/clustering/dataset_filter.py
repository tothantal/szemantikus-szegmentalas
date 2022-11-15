#!/usr/bin/python3
# -*- coding: utf-8 -*-

"""
Created on 2022. 09. 03. 18:22
@author  : A. Tóth, Sz. Szeghalmy

Egy részsorozat kinyerése a NYU_v2_depth adathalmazból.

Megjegyzés:
A projektem során csak a 'living_room' jelenetekkel kívántunk dolgozni.
Kimeneti fájlok: train.pickle, test.pickle, info_label.pickel

"""

import pickle
import numpy as np
import cv2
from pymatreader import read_mat

def show_labels(img, title = "Labels", wait = 0):
    img2 = img.astype(np.uint8)
    cv2.imshow(title, img2)
    cv2.waitKey(wait)

def show_depth(depth, title = "Depth", wait = 0):
    depth_norm = cv2.normalize(depth, 0, 255, cv2.NORM_MINMAX)
    cv2.imshow(title, depth_norm)
    cv2.waitKey(wait)

def create_filtered_data(n_train = 40,
                         n_test = 20,
                         sceneType = 'living_room',
                         fname_train = 'train.pickle',
                         fname_test = 'test.pickle',
                         fname_info = 'label_info.pickle'):
    data = read_mat('nyu_depth_v2_labeled.mat')
    print(data.keys())

    images = data['images']
    labels = data['labels']
    depths = data['depths']
    scenes_type = data['sceneTypes']

    names = data['names']
    namesToIds = data['namesToIds']

    for_train = []
    for_test = []
    for i in range(images.shape[-1]):
        if scenes_type[i] == sceneType:
            if len(for_train) < n_train:
                for_train.append( (images[:,:,:, i], depths[:,:, i], labels[:,:, i]) )
            elif len(for_test) < n_test:
                for_test.append( (images[:,:,:, i], depths[:,:, i], labels[:,:, i]) )
            else:
                break

    pickle.dump(for_train, open(fname_train, 'wb'))
    pickle.dump(for_test, open(fname_test, 'wb'))
    pickle.dump([names, namesToIds], open(fname_info, 'wb'))
#    return for_train, for_test, names, namesToIds

def load_train_files(fname = 'train.pickle'):
    """ list of (rgb [:,:,3], depth [:,:], label [:,:]) image tuples """
    return pickle.load(open(fname, 'rb'))

def load_test_files(fname = 'test.pickle'):
    """ list of (rgb [:,:,3], depth [:,:], label [:,:]) image tuples """
    return pickle.load(open(fname, 'rb'))

def load_info():
    names, namesToIds = pickle.load(open('label_info.pickle', 'rb'))
    return names, namesToIds

def test_dataset_part(fname = 'train.pickle'):
    data = load_train_files(fname)
    for img, depth, label_im in data:
        show_depth(depth, wait = 1)
        show_labels(label_im, wait=1)
        cv2.imshow("img", img)
        cv2.waitKey(10)
    cv2.waitKey(0)


if __name__ == "__main__":
    create_filtered_data()

    test_dataset_part('train.pickle')
    test_dataset_part('test.pickle')

