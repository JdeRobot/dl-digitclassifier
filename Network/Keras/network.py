#!/usr/bin/env python

"""
network.py: Train and test a CNN against an augmented MNIST dataset.

Based on https://github.com/RoboticsURJC-students/2016-tfg-david-pascual
"""
__author__ = "David Pascual Hernandez"
__date__ = "2017/03/12"

import os

import numpy as np
import tensorflow as tf
from keras.utils import vis_utils
from keras.callbacks import EarlyStopping
from keras.callbacks import ModelCheckpoint
from keras.layers import Conv2D, MaxPooling2D
from keras.layers import Dense, Dropout, Flatten
from keras.models import Sequential, load_model
from keras import backend

from DataManager.netdata import NetData
from CustomEvaluation.customevaluation import CustomEvaluation
from CustomEvaluation.customcallback import LearningCurves

import cv2

# Seed for the computer pseudo-random number generator.
np.random.seed(123)

class Network:
    def __init__(self, path):
        """
        Load the Keras model.
        @param path: str - model path
        """
        self.model = load_model(path)
        self.model._make_predict_function()
        self.graph = tf.get_default_graph()  # not quite sure why this works

        self.input_image = None
        self.output_digit = None

        self.activated = True

    def predict(self):
        """
        Classify a given digit.
        :param im: np.array - image containing a digit
        :return: classified digit
        """
        # Reshape image to fit model depending on backend

        if self.input_image is not None:
            im = self.input_image
        else:
            im = np.zeros([28, 28])


        if backend.image_dim_ordering() == 'th':
            im = im.reshape(1, 1, im.shape[0], im.shape[1])
        else:
            im = im.reshape([1, 28, 28, 1])

        with self.graph.as_default():
            self.output_digit = np.argmax(self.model.predict(im))



    def setInputImage(self, im):
        im_crop = im[140:340, 220:420]
        im_gray = cv2.cvtColor(im_crop, cv2.COLOR_BGR2GRAY)
        im_blur = cv2.GaussianBlur(im_gray, (5, 5), 0)

        im_res = cv2.resize(im_blur, (28,28))

        # Edge extraction
        im_sobel_x = cv2.Sobel(im_res, cv2.CV_32F, 1, 0, ksize=5)
        im_sobel_y = cv2.Sobel(im_res, cv2.CV_32F, 0, 1, ksize=5)
        im_edges = cv2.add(abs(im_sobel_x), abs(im_sobel_y))
        im_edges = cv2.normalize(im_edges, None, 0, 255, cv2.NORM_MINMAX)
        im_edges = np.uint8(im_edges)

        self.input_image = im_edges

        return im_edges
    

    def getOutputDigit(self):
        return self.output_digit


if __name__ == "__main__":
    nb_epoch = 100
    batch_size = 128
    nb_classes = 10
    labels = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]

    im_rows, im_cols = 28, 28
    nb_filters = 32
    kernel_size = (3, 3)
    pool_size = (2, 2)

    verbose = 0
    training = 0
    while training != "y" and training != "n":
        training = raw_input("Do you want to train the model?(y/n)")
    while verbose != "y" and verbose != "n":
        verbose = raw_input("Verbose?(y/n)")

    data = NetData(im_rows, im_cols, nb_classes, verbose)

    if training == "y":
        dropout = 0
        while dropout != "y" and dropout != "n":
            dropout = raw_input("Dropout?(y/n)")
        train_ds = raw_input("Train dataset path: ")
        while not os.path.isfile(train_ds):
            train_ds = raw_input("Enter a valid path: ")
        val_ds = raw_input("Validation dataset path: ")
        while not os.path.isfile(val_ds):
            val_ds = raw_input("Enter a valid path: ")

    test_ds = raw_input("Test dataset path: ")
    while not os.path.isfile(test_ds):
        test_ds = raw_input("Enter a valid path: ")

    if training == "y":
        # We load and reshape data in a way that it can work as input of
        # our model.
        (X_train, Y_train) = data.load(train_ds)
        (x_train, y_train), input_shape = data.adapt(X_train, Y_train)

        (X_val, Y_val) = data.load(val_ds)
        (x_val, y_val), input_shape = data.adapt(X_val, Y_val)

        # We add layers to our model.
        model = Sequential()
        model.add(Conv2D(16, kernel_size, activation="relu",
                         input_shape=input_shape))
        model.add(Conv2D(nb_filters, kernel_size, activation="relu"))
        model.add(MaxPooling2D(pool_size=pool_size))
        if dropout == "y":
            model.add(Dropout(0.25))
        model.add(Flatten())
        model.add(Dense(128, activation="relu"))
        if dropout == "y":
            model.add(Dropout(0.5))
        model.add(Dense(nb_classes, activation="softmax"))

        # We compile our model.
        model.compile(loss="categorical_crossentropy", optimizer="adadelta",
                      metrics=["accuracy"])

        # We train the model and save data to plot a learning curve.
        learning_curves = LearningCurves()
        early_stopping = EarlyStopping(monitor='val_loss', patience=2)
        checkpoint = ModelCheckpoint("net.h5", verbose=1, monitor='val_loss',
                                     save_best_only=True)

        validation = model.fit(x_train, y_train,
                               batch_size=batch_size,
                               epochs=nb_epoch,
                               validation_data=(x_val, y_val),
                               callbacks=[learning_curves, early_stopping,
                                          checkpoint])
        vis_utils.plot_model(model, "net.png", show_shapes=True)

    # We load and reshape test data and model.
    (X_test, Y_test) = data.load(test_ds)
    (x_test, y_test), input_shape = data.adapt(X_test, Y_test)

    model = load_model("net.h5")

    score = model.evaluate(x_test, y_test, verbose=0)
    print("Test score:", score[0])
    print("Test accuracy:", score[1])

    # We log the results.
    y_proba = model.predict_proba(x_test, batch_size=batch_size, verbose=1)
    y_test = np.argmax(y_test, axis=1)

    if training == "n":
        results = CustomEvaluation(y_test, y_proba, training)
    else:
        train_loss = learning_curves.loss
        train_acc = learning_curves.accuracy
        val_loss = validation.history["val_loss"]
        val_acc = validation.history["val_acc"]
        results = CustomEvaluation(y_test, y_proba, training, train_loss,
                                   train_acc, val_loss, val_acc)

    results_dict = results.dictionary()
    results.log(results_dict)

