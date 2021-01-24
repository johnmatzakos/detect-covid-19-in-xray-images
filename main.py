# Author: John Matzakos

import os
import numpy as np
import tensorflow
import keras
from keras.layers import *
from keras.models import *
from keras.preprocessing import image
from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns

#Defining paths
TRAIN_PATH = "data/train"
TEST_PATH = "data/test"

#Training model
model = Sequential()
model.add(Conv2D(32,kernel_size=(3,3),activation='relu',input_shape=(224,224,3)))
model.add(Conv2D(128,(3,3),activation='relu'))
model.add(MaxPooling2D(pool_size=(2,2)))
model.add(Dropout(0.25))

model.add(Conv2D(64,(3,3),activation='relu'))
model.add(MaxPooling2D(pool_size=(2,2)))
model.add(Dropout(0.25))

model.add(Conv2D(128,(3,3),activation='relu'))
model.add(MaxPooling2D(pool_size=(2,2)))
model.add(Dropout(0.25))

model.add(Flatten())
model.add(Dense(64,activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(1,activation='sigmoid'))

model.compile(loss=keras.losses.binary_crossentropy,optimizer='adam',metrics=['accuracy'])

#Getting parameters
model.summary()

#Moulding train images
train_datagen = image.ImageDataGenerator(rescale = 1./255, shear_range = 0.2,zoom_range = 0.2, horizontal_flip = True)

test_dataset = image.ImageDataGenerator(rescale=1./255)

#Reshaping test and validation images
train_generator = train_datagen.flow_from_directory(
    'data/train',
    target_size = (224,224),
    batch_size = 32,
    class_mode = 'binary')
validation_generator = test_dataset.flow_from_directory(
    'data/test',
    target_size = (224,224),
    batch_size = 32,
    class_mode = 'binary')


#Training the model
hist_new = model.fit_generator(
    train_generator,
    steps_per_epoch=8,
    epochs = 10,
    validation_data = validation_generator,
    validation_steps=2
)

#Getting summary
# summary=hist.history
# print(summary)

# model.save("model_covid.h5")

model.evaluate_generator(train_generator)

print(model.evaluate_generator(validation_generator))


train_generator.class_indices

y_actual, y_test = [],[]

for i in os.listdir("./data/test/Normal/"):
    img=image.load_img("./data/test/Normal/"+i,target_size=(224,224))
    img=image.img_to_array(img)
    img=np.expand_dims(img,axis=0)
    pred=model.predict_classes(img)
    y_test.append(pred[0,0])
    y_actual.append(1)

for i in os.listdir("./data/test/Covid/"):
    img=image.load_img("./data/test/Covid/"+i,target_size=(224,224))
    img=image.img_to_array(img)
    img=np.expand_dims(img,axis=0)
    pred=model.predict_classes(img)
    y_test.append(pred[0,0])
    y_actual.append(0)

y_actual=np.array(y_actual)
y_test=np.array(y_test)


cn=confusion_matrix(y_actual,y_test)

sns.heatmap(cn,cmap="plasma",annot=True) #0: Covid ; 1: Normal

