# Detect Covid-19 In X-Ray Images

This projects aims to detect Covid-19 using Deep Learning in x-ray images. Initially, VGG-16 convolutional neural network is used to perform this classification task. In the future the perfromance of other neural networks will be studied.

### Python libraries used:
- Tensorflow
- Keras
- Open CV
- Scikit Learn
- Numpy
- Matplotlib

### Dataset sources:
- For x-ray images labeled 'covid':   https://github.com/ieee8023/covid-chestxray-dataset by Dr. Joseph Cohen
- For x-ray images labeled 'normal':  https://www.kaggle.com/paultimothymooney/chest-xray-pneumonia

### Results:
- Accuracy:     0.90
- Sensitivity:  0.80
- Specificity:  1.00
- Classification report:

              precision    recall  f1-score   support

       covid       1.00      0.80      0.89     5
       
      normal       0.83      1.00      0.91     5
      
      accuracy                         0.90     10
      macro avg    0.92      0.90      0.90     10
      weighted avg 0.92      0.90      0.90     10
  
### Installation of dependencies:

```
pip install -r installation/requirements.txt
```
