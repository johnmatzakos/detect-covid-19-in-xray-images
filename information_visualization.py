# Author: John Matzakos

import matplotlib.pyplot as plt
import numpy as np
from keras.utils.vis_utils import plot_model


def execute_visualization(epochs, trained_model):
    # plot the training loss and accuracy
    plt.style.use("ggplot")
    plt.figure()
    plt.plot(np.arange(0, epochs), trained_model.history["loss"], label="train_loss")
    plt.plot(np.arange(0, epochs), trained_model.history["val_loss"], label="val_loss")
    plt.plot(np.arange(0, epochs), trained_model.history["accuracy"], label="train_acc")
    plt.plot(np.arange(0, epochs), trained_model.history["val_accuracy"], label="val_acc")
    plt.title("Training Loss and Accuracy on COVID-19 Dataset")
    plt.xlabel("Epoch #")
    plt.ylabel("Loss/Accuracy")
    plt.legend(loc="lower left")
    plt.savefig("plot")


def plot(model):
    """
    Plots the architecture of a convolutional neural network.
    :param model: convolutional neural network
    :return: image
    """
    # Plot the layers of VGG-16
    plot_model(model, to_file='vgg.png')


