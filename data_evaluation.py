# Author: John Matzakos

import os
import numpy as np
from keras.preprocessing import image
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
import seaborn as sns


def evaluate(model, train_generator, test_generator, test_set_path):
    model.evaluate(train_generator)

    print(model.evaluate(test_generator))

    print(train_generator.class_indices)

    y_actual, y_test = [], []

    for i in os.listdir("./"+test_set_path+"/Normal/"):
        img = image.load_img("./"+test_set_path+"/Normal/"+i, target_size=(224, 224))
        img = image.img_to_array(img)
        img = np.expand_dims(img, axis=0)
        pred = model.predict_classes(img)
        y_test.append(pred[0, 0])
        y_actual.append(1)

    for i in os.listdir("./"+test_set_path+"/Covid/"):
        img = image.load_img("./"+test_set_path+"/Covid/"+i, target_size=(224, 224))
        img = image.img_to_array(img)
        img = np.expand_dims(img, axis=0)
        pred = model.predict_classes(img)
        y_test.append(pred[0, 0])
        y_actual.append(0)

    y_actual = np.array(y_actual)
    y_test = np.array(y_test)

    report = classification_report(y_actual, y_test, target_names=['Covid', 'Normal'],  zero_division=0)
    print(report)

    cn = confusion_matrix(y_actual, y_test)

    # 0: Covid | 1: Normal
    sns.heatmap(cn, cmap="plasma", annot=True)
