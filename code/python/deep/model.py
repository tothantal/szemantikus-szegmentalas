# -*- coding: utf-8 -*-
"""
Created on Tue Nov 22 13:20:11 2022

@author: Tóth Antal
"""

import tensorflow as tf

def create_model():
    model = tf.keras.Sequential([
            tf.keras.Input(shape = (480, 640, 3)),
            tf.keras.layers.Conv2D(16, (3, 3)),
            tf.keras.layers.BatchNormalization(),
            tf.keras.layers.MaxPooling2D((2, 2)),
            tf.keras.layers.Conv2D(16, (3, 3)),
            tf.keras.layers.BatchNormalization(),
            tf.keras.layers.MaxPooling2D((2, 2)),
            tf.keras.layers.UpSampling2D((2, 2)),
            tf.keras.layers.Conv2DTranspose(16, (3, 3)),
            tf.keras.layers.BatchNormalization(),
            tf.keras.layers.UpSampling2D((2, 2)),
            tf.keras.layers.Conv2DTranspose(1, (5, 5))
        ])
    
    return model

if __name__ == "__main__":
    model = create_model()
