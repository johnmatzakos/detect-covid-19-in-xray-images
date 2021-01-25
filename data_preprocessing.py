# Author: John Matzakos

from keras.preprocessing import image

# Defining paths
training_set_path = "data/train"
test_set_path = "data/test"


# Moulding train images
def train_data_moulding():
    return image.ImageDataGenerator(rescale=1./255, shear_range=0.2, zoom_range=0.2, horizontal_flip=True)


# Moulding train images
def test_data_moulding():
    return image.ImageDataGenerator(rescale=1./255)


# Reshaping train and test images
def train_generator(train_datagen):
    return train_datagen.flow_from_directory(
        training_set_path,
        target_size=(224, 224),
        batch_size=32,
        class_mode='binary')


def test_generator(test_datagen):
    return test_datagen.flow_from_directory(
        test_set_path,
        target_size=(224, 224),
        batch_size=32,
        class_mode='binary')
