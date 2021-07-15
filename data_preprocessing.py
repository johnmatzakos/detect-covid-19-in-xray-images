# Author: John Matzakos

import os

import cv2
import numpy as np
from imutils import paths
from sklearn.preprocessing import LabelBinarizer
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.utils import to_categorical

from utilities import logger

log = logger.setup_logger("data-preprocessing")


def execute_data_preprocessing(dataset_path, height, width):
    (data, labels) = load_swap_resize(dataset_path, height, width)

    # perform one-hot encoding on the labels
    (lb, labels) = one_hot_encoding(labels)

    # convert the data and labels to NumPy arrays while scaling the pixel intensities to the range [0, 1]
    (data, labels) = image_to_numpy_array(data, labels)

    log.info("Data preprocessing phase executed.")

    return data, labels, lb


def load_swap_resize(dataset_path, height, width):
    log.info("Loading images...")
    # get the list of images in the dataset directory
    image_paths = list(paths.list_images(dataset_path))
    data = []
    labels = []

    for image_path in image_paths:
        # extract the class label from the filename
        label = image_path.split(os.path.sep)[-2]
        # load the image
        image = cv2.imread(image_path)
        # swap color channels
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        # resize it to be a fixed at 224x224 pixels while ignoring aspect ratio
        image = cv2.resize(image, (height, width))
        # update the data and labels lists, respectively
        data.append(image)
        labels.append(label)

    return data, labels


def image_to_numpy_array(data, labels):
    """
    Convert the data and labels to NumPy arrays while scaling the pixel intensities to the range [0, 1]
    :param data:
    :param labels:
    :return:
    """
    data = np.array(data) / 255.0
    labels = np.array(labels)
    return data, labels


def one_hot_encoding(labels):
    """
    Performs one-hot encoding to the labels
    :param labels:
    :return: labels
    """
    lb = LabelBinarizer()
    labels = lb.fit_transform(labels)
    labels = to_categorical(labels)
    return lb, labels


def data_augmentator(rotation_range, fit_model):
    """
    Performs data augmentation
    :param rotation_range:
    :param fit_model:
    :return:
    """
    return ImageDataGenerator(rotation_range=rotation_range, fill_mode=fit_model)
