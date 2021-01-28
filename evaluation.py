# Author: John Matzakos

import numpy as np
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix


def execute_evaluation(model, testX, testY, batch_size, lb):
    # make predictions on the testing set
    print("Evaluating the neural network...")
    predIdxs = model.predict(testX, batch_size=batch_size)
    # for each image in the testing set we need to find the index of the
    # label with corresponding largest predicted probability
    predIdxs = np.argmax(predIdxs, axis=1)
    # show a nicely formatted classification report

    report = classification_report(testY.argmax(axis=1), predIdxs, target_names=lb.classes_)
    print(report)
    # compute the confusion matrix and and use it to derive the raw
    # accuracy, sensitivity, and specificity
    cm = confusion_matrix(testY.argmax(axis=1), predIdxs)
    print(cm)
    total = sum(sum(cm))
    acc = (cm[0, 0] + cm[1, 1]) / total
    sensitivity = cm[0, 0] / (cm[0, 0] + cm[0, 1])
    specificity = cm[1, 1] / (cm[1, 0] + cm[1, 1])

    # show the confusion matrix, accuracy, sensitivity, and specificity
    print(cm)
    print("acc: {:.4f}".format(acc))
    print("sensitivity: {:.4f}".format(sensitivity))
    print("specificity: {:.4f}".format(specificity))

