#!/usr/bin/env python

"""
digitclassifier.py: It receives images from a live video and classify
them into digits employing a convolutional neural network. It also
shows live video and results in a GUI.

Based on @nuriaoyaga code:
https://github.com/RoboticsURJC-students/2016-tfg-nuria-oyaga/blob/master/numberclassifier.py

and @dpascualhe's:
https://github.com/RoboticsURJC-students/2016-tfg-david-pascual/blob/master/digitclassifier.py
"""
__author__ = "naxvm"
__date__ = "2017/10/--"

import signal
import sys
import yaml
from PyQt5 import QtWidgets

from Camera.camera import Camera
from Camera.threadcamera import ThreadCamera
from GUI.gui import GUI
from GUI.threadgui import ThreadGUI
from Estimator.estimator import Estimator
from Estimator.threadestimator import ThreadEstimator

signal.signal(signal.SIGINT, signal.SIG_DFL)

if __name__ == '__main__':
    # Parse YAML config. file
    data = None
    with open(sys.argv[1], "r") as stream:
        try:
            data = yaml.load(stream)
        except yaml.YAMLError as exc:
            print(exc)

    cam = Camera()
    app = QtWidgets.QApplication(sys.argv)
    window = GUI(cam)
    estimator = Estimator(window, data)
    window.show()

    # Threading camera
    t_cam = ThreadCamera(cam)
    t_cam.start()

    # Threading estimator
    t_estimator = ThreadEstimator(estimator)
    t_estimator.start()

    # Threading GUI
    t_gui = ThreadGUI(window)
    t_gui.start()

    sys.exit(app.exec_())
