#!/usr/bin/env python

"""
estimator.py: Estimator class.
"""
__author__ = "David Pascual Hernandez"
__date__ = "2017/11/16"

import cv2
import numpy as np

class Estimator:
    """
    Estimator class makes a prediction given an input image.
    """
    def __init__(self, gui, data):
        """
        Estimator object constructor.
        @param gui: GUI object
        @param data: parsed YAML config. file
        """
        self.gui = gui
        self.cam = self.gui.cam

        # Import only required network
        available_fw = ["Keras", "Tensorflow"]
        framework = data["DigitClassifier"]["Framework"]
        if framework == "Keras":
            from Estimator.Keras.network import Network
        elif framework == "TensorFlow":
            from Estimator.TensorFlow.network import Network
        else:
            print("'%s' framework is not supported. Please choose one of: %s"
                  % (framework, ', '.join(available_fw)))
            exit()

        # Load model
        model_path = data["DigitClassifier"]["Model"]
        self.net = Network(model_path)

        # Print info.
        print("\nFramework: %s\nModel: %s\nImage size: %dx%d px"
              % (framework, model_path, self.cam.im.width, self.cam.im.height))

    def estimate(self, im):
        """
        Predict the digit present in the image.
        @param im: np.array - input image
        @return: str - predicted digit
        """
        
        return self.net.predict(im)

    @staticmethod
    def transform_image(im):
        """
        Transform the image before prediction.
        @param im: np.array - input image
        @return: np.array - transformed image
        """
        im_crop = im[140:340, 220:420]
        im_gray = cv2.cvtColor(im_crop, cv2.COLOR_BGR2GRAY)
        im_blur = cv2.GaussianBlur(im_gray, (5, 5), 0)  # Noise reduction.

        im_res = cv2.resize(im_blur, (28, 28))

        # Edge extraction.
        im_sobel_x = cv2.Sobel(im_res, cv2.CV_32F, 1, 0, ksize=5)
        im_sobel_y = cv2.Sobel(im_res, cv2.CV_32F, 0, 1, ksize=5)
        im_edges = cv2.add(abs(im_sobel_x), abs(im_sobel_y))
        im_edges = cv2.normalize(im_edges, None, 0, 255, cv2.NORM_MINMAX)
        im_edges = np.uint8(im_edges)

        return im_edges

    def update(self):
        """
        Update estimator.
        """
        im = self.cam.get_image()
        im_trans=self.transform_image(im)
        digit = self.estimate(im_trans)

        self.gui.pred = digit
        self.gui.im_trans = im_trans
