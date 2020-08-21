"""
@author: idankri
python version: 3.7
tensorflow version: 2.3.0
pandas version: 1.0.1
numpy version: 1.18.1

Builds a model that receives a 28 x 28 pixel array and try to guess the number that is
written in the picture
"""
# imports
import pandas as pd
import numpy as np
import tensorflow as tf
from src.GlobalFields import *
import os, sys
from src.ImageParser import ImageParser
import logging, json

log = logging.getLogger(__name__)

# default model path, can be changed to any path
DEFAULT_MODEL_PATH = r"../res/my_model"

# Disable warning and info logs printed by TensorFlow
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'


class NumberDetectorModel(metaclass=Singleton):
    def __init__(self, path: str = None, training_data: TrainingData = None):
        self.path = path if path else DEFAULT_MODEL_PATH
        if os.path.exists(self.path):
            # load model and training data (hyper-parameters)
            self.model = tf.keras.models.load_model(self.path)
            self.training_data = training_data
            if not training_data:
                with open(f"{self.path}.json", "r") as f:
                    training_data_dict = json.load(f)
                self.training_data = TrainingData(**training_data_dict)
        else:
            # create a new model and train it
            temp = sys.stdout
            sys.stdout = LoggerWriter(log.info)
            self.training_data = training_data
            train_x, train_y, test_x, test_y = self.prepare_data()
            self.create_model()
            self.train_model(train_x, train_y)
            sys.stdout = temp
            self.evaluate_model(test_x, test_y)
            # save in path
            self.model.save(self.path)
            with open(f"{self.path}.json", "w") as f:
                json.dump(self.training_data._asdict(), f)

    ##### properties #####

    @property
    def training_data(self):
        return self._training_data

    @training_data.setter
    def training_data(self, value: TrainingData):
        self._training_data = value

    @property
    def learning_rate(self):
        return self.training_data.learning_rate

    @property
    def epochs(self):
        return self.training_data.epochs

    @property
    def batch_size(self):
        return self.training_data.batch_size

    @property
    def validation_split(self):
        return self.training_data.validation_split

    ##### end of properties #####

    @staticmethod
    def prepare_data():
        """
        Loads keras mnist dataset and normalize it
        :return: Normalized sets of mnist train + test labels and features
        """
        (train_x, train_y), (test_x, test_y) = tf.keras.datasets.mnist.load_data()
        # normalize
        train_x = train_x / 255.0
        test_x = test_x / 255.0

        return (train_x, train_y, test_x, test_y)

    def create_model(self):
        """
        Creates a neural network fit to be used as number predictor
        From my personal experiments yields approximately 97~98% accuracy
        :return: The Keras model (also saved as an attribute)
        """
        model = tf.keras.models.Sequential()
        model.add(tf.keras.layers.Flatten(input_shape=(28, 28,)))
        model.add(tf.keras.layers.Dense(units=128, activation='relu'))
        model.add(tf.keras.layers.Dropout(rate=0.4))
        model.add(tf.keras.layers.Dense(units=10, activation='softmax'))

        model.compile(optimizer=tf.keras.optimizers.Adam(lr=self.learning_rate),
                      loss="sparse_categorical_crossentropy",
                      metrics=['accuracy'])
        self.model = model

        return model

    def train_model(self, train_x, train_y):
        """
        Trains the model with a set of features and set of labels
        :param train_x: set of features
        :param train_y: set of labels
        :return: history for training the model
        """
        history = self.model.fit(x=train_x, y=train_y, epochs=self.epochs,
                                 batch_size=self.batch_size, shuffle=True,
                                 validation_split=self.validation_split)

        return (history.epoch, pd.DataFrame(history.history))

    def evaluate_model(self, test_x, test_y):
        """
        Tests the model with set of test features and labels
        :param test_x: test features
        :param test_y: test labels
        :return: None. Evaluation will be printed to stdout
        """
        self.model.evaluate(x=test_x, y=test_y, batch_size=self.batch_size)

    def predict(self, parsed_image):
        """
        Predicts result for a parsed image (preferably by Image Parser utility)
        :param parsed_image: numpy array of 28 x 28 numpy arrays
        :return: Predicted number
        """
        prediction_list = self.model.predict(parsed_image)
        return list(prediction_list[0]).index(prediction_list.max())


if __name__ == '__main__':
    img = ImageParser.parse_28_by_28_png_picture("2_2.png")
    # declare hyper-parameters
    # learning_rate = 0.003
    # epochs = 50
    # batch_size = 4000
    # validation_split = 0.2

    trained_model = NumberDetectorModel()
    prediction = trained_model.predict(img)
    print(f"prediction is: {prediction}")
