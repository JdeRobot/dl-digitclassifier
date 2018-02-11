#!/usr/bin/env python

"""
camera.py: Class which abstracts a Camera from a proxy (created by
ICE/ROS), and provides the methods to keep it constantly updated.
Also, it processes the images acquired by using a Sobel edges filter,
and delivers it to the neural network, which returns the predicted
digit.

Based on @nuriaoyaga code:
https://github.com/RoboticsURJC-students/2016-tfg-nuria-oyaga/blob/master/numberclassifier.py

and @dpascualhe"s:
https://github.com/RoboticsURJC-students/2016-tfg-david-pascual/blob/master/digitclassifier.py

Based on:
https://github.com/RoboticsURJC-students/2016-tfg-david-pascual
"""
__author__ = "naxvm"
__date__ = "-"

import sys
import threading
import traceback
import yaml

import comm
import config
import cv2
import numpy as np


class Camera:
    """
    Camera class gets images from live video and transform them
    in order to predict the digit in the image.
    """

    def __init__(self):
        """
        Camera object constructor
        @param data: parsed YAML config. file
        """
        # Creation of the Camera through the comm-ICE proxy.
        try:
            cfg = config.load(sys.argv[1])
        except IndexError:
            raise SystemExit("Error: Missing YML file. \n Usage: python2"
                             "digitclassifier.py digitclassifier.yml")

        jdrc = comm.init(cfg, "DigitClassifier")

        self.lock = threading.Lock()

        # Acquire first image
        try:
            self.cam = jdrc.getCameraClient("DigitClassifier.Camera")

            if self.cam.hasproxy():
                self.im = self.cam.getImage()
            else:
                print("Interface camera not connected")
                exit()

        except:
            traceback.print_exc()
            exit()

    def get_image(self):
        """
        Get an image from the webcam and return the original image
        with a ROI draw over it and the transformed image that we're
        going to use to make the prediction.
        @return: list - original & transformed images
        """
        if self.cam:
            self.lock.acquire()

            im = np.frombuffer(self.im.data, dtype=np.uint8)
            im.shape = self.im.height, self.im.width, 3
            im_trans = self.transform_image(im)

            cv2.rectangle(im, (218, 138), (422, 342), (0, 0, 255), 2)
            ims = [im, im_trans]

            self.lock.release()

            return ims

    def update(self):
        """
        Update Camera every time the thread changes.
        """
        if self.cam:
            self.lock.acquire()

            self.im = self.cam.getImage()

            self.lock.release()

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
