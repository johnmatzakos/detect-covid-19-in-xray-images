# Author: John Matzakos

import keras
from keras.layers import *
from keras.models import *
from tensorflow.keras.applications import VGG16
from tensorflow.keras.layers import AveragePooling2D
from tensorflow.keras.layers import Dense
from tensorflow.keras.layers import Dropout
from tensorflow.keras.layers import Flatten
from tensorflow.keras.layers import Input
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam

from data_preprocessing import data_augmentator
from utilities import logger

log = logger.setup_logger("data-mining")


def custom_cnn():
    model = Sequential()
    model.add(Conv2D(32, kernel_size=(3, 3), activation='relu', input_shape=(224, 224, 3)))
    model.add(Conv2D(128, (3, 3), activation='relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(0.25))

    model.add(Conv2D(64, (3, 3), activation='relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(0.25))

    model.add(Conv2D(128, (3, 3), activation='relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(0.25))

    model.add(Flatten())
    model.add(Dense(64, activation='relu'))
    model.add(Dropout(0.5))
    model.add(Dense(1, activation='sigmoid'))

    model.compile(loss=keras.losses.binary_crossentropy, optimizer='adam', metrics=['accuracy'])

    return model


def vgg_16(weights, include_top, height, width, number_of_channels, number_of_classes):
    """
    Loads the VGG-16 Convolutional Neural Network
    :param weights: string, such as 'imagenet'
    :param include_top: boolean
    :param height: integer
    :param width: integer
    :param number_of_channels: integer
    :param number_of_classes: integer
    :return: VGG16 object (convolutional neural network)
    """
    """
     Model Arguments:
    - weights (‘imagenet‘): What weights to load. You can specify None to not load pre-trained weights
        if you are interested in training the model yourself from scratch.
    - include_top (True): Whether or not to include the output layers for the model.
        You don’t need these if you are fitting the model on your own problem.
    - input_tensor (None): A new input layer if you intend to fit the model on new data of a different size.
    - input_shape (None): The size of images that the model is expected to take if you change the input layer.
    - pooling (None): The type of pooling to use when you are training a new set of output layers.
    - classes (1000): The number of classes (e.g. size of output vector) for the model.
    """
    # Load the pre-trained model of VGG-16 CNN
    model = VGG16(weights=weights, include_top=include_top,
                  input_tensor=Input(shape=(height, width, number_of_channels)), classes=number_of_classes)

    log.info("Loaded VGG16 Convolutional Neural Network.")

    return model


def get_model_head(model):
    """
    Construct the head of the model that will be placed on top of the base model
    :param model: convolutional neural network
    :return: modified convolutional neural network
    """
    headModel = model.output
    headModel = AveragePooling2D(pool_size=(4, 4))(headModel)
    headModel = Flatten(name="flatten")(headModel)
    headModel = Dense(64, activation="relu")(headModel)
    headModel = Dropout(0.5)(headModel)
    headModel = Dense(2, activation="softmax")(headModel)
    # place the head model on top of the base model
    model = Model(inputs=model.input, outputs=headModel)

    log.info("Constructed a new model head for the VGG16 Convolutional Neural Network.")

    return model


def freeze_layers(model):
    """
    Freezes all layers in the base model in order not to be updated during the first training process
    :param model: convolutional neural network
    :return: convolutional neural network with freezed layers
    """
    for layer in model.layers:
        layer.trainable = False

    log.info("Froze all layers of the base model.")

    return model


def compile_model(model, learning_rate, epochs):
    """
    Compiles the constructed model in order to be trained.
    :param model: convolutional neural network
    :param learning_rate: integer
    :param epochs: integer
    :return: compiled convolutional neural network
    """
    print("Compiling model...")
    opt = Adam(lr=learning_rate, decay=learning_rate / epochs)

    model.compile(loss=keras.losses.binary_crossentropy, optimizer=opt, metrics=["accuracy"])

    log.info("Compiled VGG16 Convolutional Neural Network.")

    return model


# Training the model
def train_model(model, trainX, trainY, testX, testY, batch_size, epochs):
    log.info("Training head model...")

    rotation_range = 15
    fit_model = "nearest"
    aug = data_augmentator(rotation_range, fit_model)

    return model.fit(
        aug.flow(trainX, trainY, batch_size=batch_size),
        steps_per_epoch=len(trainX) // batch_size,
        validation_data=(testX, testY),
        validation_steps=len(testX) // batch_size,
        epochs=epochs
    )
