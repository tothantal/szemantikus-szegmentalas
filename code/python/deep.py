"""

@author: Tóth Antal

"""

from tensorflow.keras.utils import to_categorical
from sklearn.model_selection import train_test_split
import numpy as np
import keras.backend as K
import tensorflow as tf
import tensorflow_datasets as tfds

# Loading Data
ds, info = tfds.load('nyu_depth_v2',
                     data_dir = "d:/tensor/",
                     split = ['train[1%:]', 'train[:1%]'], 
                     as_supervised = True, 
                     with_info = True,
                     shuffle_files = True)

ds_train = ds[1]

# Extract numpy.array X & y from tf.data.Dataset
X_numpy = np.asarray(list(map(lambda x: x[0], tfds.as_numpy(ds_train))))
y_numpy = np.asarray(list(map(lambda x: x[1], tfds.as_numpy(ds_train))))
X_train, X_test, y_train, y_test = train_test_split(X_numpy, y_numpy, 
                                                    test_size=0.2, 
                                                    random_state=42)
y_train = to_categorical(y_train, 20)
y_test = to_categorical(y_test, 20)

# Creating Model
model = tf.keras.Sequential([
    tf.keras.layers.Dense(3, activation = tf.nn.relu, input_shape = (480, 640, 3)),    
    tf.keras.layers.Dense(10, activation = tf.nn.relu),
    tf.keras.layers.Dense(20, activation = 'softmax')
])

# F1
def f1_metric(y_true, y_pred):
    true_positives = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
    possible_positives = K.sum(K.round(K.clip(y_true, 0, 1)))
    predicted_positives = K.sum(K.round(K.clip(y_pred, 0, 1)))
    precision = true_positives / (predicted_positives + K.epsilon())
    recall = true_positives / (possible_positives + K.epsilon())
    f1_val = 2 * (precision * recall) / (precision+recall + K.epsilon())
    return f1_val

# Compiling
jacard =  tf.keras.metrics.MeanIoU(num_classes = 30)
model.compile(loss = 'categorical_crossentropy', 
              optimizer='adam', 
              metrics=['accuracy', jacard, f1_metric])

# Summary
model.summary()

# Training
model.fit(X_train, y_train, epochs = 10, 
          batch_size = 1, validation_data=(X_test, y_test))
