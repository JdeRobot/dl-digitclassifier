#
# Created on Mar 7, 2017
#
# @author: dpascualhe
#
# Based on @nuriaoyaga code:
# https://github.com/RoboticsURJC-students/2016-tfg-nuria-oyaga/blob/
#     master/camera/camera.py
#
# And @Javii91 code:
# https://github.com/Javii91/Domotic/blob/master/Others/cameraview.py
#

import sys
import threading
import traceback

import comm
import config
import cv2
import numpy as np
from Net.network import Network

class Camera:

    def __init__(self):
        ''' Camera class gets images from live video and transform them
        in order to predict the digit in the image.
        '''
        print "\nLoading Keras model..."
        self.net = Network("Net/Model/net_4conv_patience5.h5")
        print "loaded\n"

        try:
            cfg = config.load(sys.argv[1])
        except IndexError:
            raise SystemExit('Error: Missing YML file. \n  Usage: python2 digitclassifier.py digitclassifier.yml')

        # starting comm
        jdrc = comm.init(cfg, 'DigitClassifier')


        self.lock = threading.Lock()

        try:
            self.cam = jdrc.getCameraClient("DigitClassifier.Camera")
            if self.cam.hasproxy():
                self.im = self.cam.getImage()
                self.im_height = self.im.height
                self.im_width = self.im.width
                print("Camera succesfully connected!")

        except:
            traceback.print_exc()
            exit()

    def getImage(self):
        ''' Gets the image from the webcam and returns the original
        image with a ROI draw over it and the transformed image that
        we're going to use to make the prediction.
        '''
        if self.cam.hasproxy():
            self.lock.acquire()

            im = np.frombuffer(self.im, dtype=np.uint8)
            im.shape = self.im_height, self.im_width, 3
            im_trans = self.trasformImage(im)

            # It prints the ROI over the live video
            cv2.rectangle(im, (218, 138), (422, 342), (0, 0, 255), 2)
            ims = [im, im_trans]

            self.lock.release()

            return ims

    def update(self):
        ''' Updates the camera every time the thread changes. '''
        if self.cam:
            self.lock.acquire()

            im = self.cam.getImage()

            self.im = im.data
            self.im_height = im.height
            self.im_width = im.width

            self.lock.release()

    def trasformImage(self, im):
        ''' Transforms the image into a 28x28 pixel grayscale image and
        applies a sobel filter (both x and y directions).
        '''
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

    def predict(self, im):
        """
        Classify the digit in the image.
        @param im: np.array - input image
        """

        return self.net.predict(im)

