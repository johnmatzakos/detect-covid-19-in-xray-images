# Author: John Matzakos

from sklearn.model_selection import train_test_split
import pickle


from data_mining import *
from data_preprocessing import *
from evaluation import *
from information_visualization import *
from utilities import logger
from utilities.date_utils import get_timestamp

log = logger.setup_logger("main")

# Defining dataset based on working directory
dataset_path = "dataset"

# Declare And Initialize Parameters
# basic parameters
weights = "imagenet"
include_top = False
height = 224
width = 224
number_of_channels = 3
number_of_classes = 2
# hyperparameters
learning_rate = 1e-3
epochs = 25
batch_size = 8

# Data Preprocessing Phase
(data, labels, lb) = execute_data_preprocessing(dataset_path, height, width)

# Split the dataset into training set (80%) and test set (20%)
(trainX, testX, trainY, testY) = train_test_split(data, labels, test_size=0.20, stratify=labels, random_state=42)

# Data Mining Phase
# load VGG-16 CNN model
model = vgg_16(weights, include_top, height, width, number_of_channels, number_of_classes)

# place the model head on top of the basic model
model = get_model_head(model)

# Get model summary
log.info(model.summary())

# compile the constructed model
model = compile_model(model, learning_rate, epochs)

# freeze all vgg-16 layers so only the custom model will be trained
model = freeze_layers(model)

# training the CNN Model
trained_model = train_model(model, trainX, trainY, testX, testY, batch_size, epochs)

# Evaluation Phase
execute_evaluation(model, testX, testY, batch_size, lb)

# Get current date and time in timestamp format
timestamp = get_timestamp()

# Information Visualization Phase
execute_visualization(epochs, trained_model, f"visualizations/vgg16_training_loss_and_accuracy_{timestamp}.png")
plot_ann_architecture(trained_model, f"visualizations/vgg16_{timestamp}.png")

# Serialize the model, save it for deployment
log.info("Saving model...")
model.save(f"models/covid19_vgg16_model_{timestamp}.h5", save_format="h5")

# Save the model to a .pkl file
Pkl_Filename = f"models/covid19_vgg16_model_{timestamp}.pkl"

with open(Pkl_Filename, 'wb') as file:
    pickle.dump(model, file)
