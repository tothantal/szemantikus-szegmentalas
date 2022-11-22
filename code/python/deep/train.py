# -*- coding: utf-8 -*-
"""
Created on Tue Nov 22 13:25:03 2022

@author: Tóth Antal
"""

from sklearn.model_selection import train_test_split
import numpy as np
import tensorflow as tf
import pickle

import model as md
import metrics as me

def load_train_files(fname = 'train.pickle'):
    """ list of (rgb [:,:,3], depth [:,:], label [:,:]) image tuples """
    return pickle.load(open(fname, 'rb'))

if __name__ == "__main__":
    model = md.create_model()
    data = load_train_files('train.pickle')
    
    X = []
    y = []
    for images in data:
            image = images[0]
            label = images[2]
            X.append(image)
            y.append(label)
            
    X = np.array(X)
    y = np.array(y)
    X_train, X_test, y_train, y_test = train_test_split(X, y)
    
    optimizer = tf.keras.optimizers.Adam( learning_rate = 0.00000001)
    model.compile(
        loss = 'categorical_crossentropy', 
        optimizer=optimizer, 
        metrics=['accuracy', me.f1]
    )
    
    print(model.summary())
    
    model.fit(X_train, y_train, epochs = 10, 
          batch_size = 1, validation_data=(X_test, y_test))

    pickle.dump(model, open("deepmodel.pickle", 'wb'))
