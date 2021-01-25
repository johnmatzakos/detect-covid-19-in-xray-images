# Author: John Matzakos

from data_preprocessing import *
from data_mining import *
from data_evaluation import *

# Defining paths
training_set_path = "data/train"
test_set_path = "data/test"

# CNN Architecture
model = custom_cnn()

# Getting parameters
model.summary()

# Data Preprocessing
train_datagen = train_data_moulding()
test_datagen = test_data_moulding()

train_generator = train_generator(train_datagen)
test_generator = test_generator(test_datagen)

# Data Mining: Training CNN Model
hist_new = train_model(model, train_generator, test_generator)

# Getting summary
print(hist_new.history)

# Data Evaluation
evaluate(model, train_generator, test_generator, test_set_path)

# Save model for deployment
model.save("covid-19_cnn_model.h5")
