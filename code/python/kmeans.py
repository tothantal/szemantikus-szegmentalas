"""

@author: Tóth Antal

"""

import tensorflow_datasets as tfds
import cv2
import numpy as np
import matplotlib.pyplot as plt

def load_dataset():
    return tfds.load('nyu_depth_v2', split = tfds.Split.TRAIN, shuffle_files = True)

def cluster_single_image():
    ds = load_dataset()
    
    # Get images
    ds_example, = ds.take(1)
    img = ds_example['image']

    # convert image to rgb
    rgb_image = cv2.cvtColor(img.numpy(), cv2.COLOR_BGR2RGB)

    # reshape the image to a 2D array of pixels and 3 color values
    pixel_values = rgb_image.reshape((-1, 3))

    # convert to float
    pixel_values = np.float32(pixel_values)

    # clustering criteria
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 100, 0.2)

    # number of clusters
    k = 10 

    # clustering
    _, labels, (centers) = cv2.kmeans(pixel_values, k, None, criteria, 21, cv2.KMEANS_RANDOM_CENTERS)

    # defining centers, labels
    centers = np.uint8(centers)
    labels = labels.flatten()

    # creating segmented image
    segmented_image = centers[labels.flatten()]
    segmented_image = segmented_image.reshape(rgb_image.shape)

    # printing image
    plt.axis('off')
    plt.imshow(segmented_image)
    
    
def main():
    cluster_single_image()
    
main()
