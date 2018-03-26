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

import sys
import signal

from PyQt5 import QtWidgets

from Camera.camera import Camera
from Camera.threadcamera import ThreadCamera
from GUI.gui import GUI
from GUI.threadgui import ThreadGUI
from Network.threadnetwork import ThreadNetwork

import config
import comm

signal.signal(signal.SIGINT, signal.SIG_DFL)

if __name__ == '__main__':
    # Parse YAML config. file
    try:
        cfg = config.load(sys.argv[1])
    except IndexError:
        raise SystemExit('Missing YML file. Usage: python2 digitclassifier.py digitclassifier.yml')

    # Create camera proxy
    jdrc = comm.init(cfg, 'DigitClassifier')
    proxy = jdrc.getCameraClient('DigitClassifier.Camera')

    # Parse network parameters
    network_framework = cfg.getNode()['DigitClassifier']['Framework']
    network_model_path = cfg.getNode()['DigitClassifier']['Model']

    # We define the network import depending on the chosen framework
    if network_framework.lower() == 'keras':
        from Network.Keras.network import Network
        framework_title = 'Keras'
    elif network_framework.lower() == 'tensorflow':
        from Network.TensorFlow.network import Network
        framework_title = 'TensorFlow'
    else:
        raise SystemExit(('%s not supported! Supported frameworks: Keras, TensorFlow') % (network_framework))

    cam = Camera(proxy)
    t_cam = ThreadCamera(cam)
    t_cam.start()

    network = Network(network_model_path)
    network.setCamera(cam)
    t_network = ThreadNetwork(network)
    t_network.start()

    app = QtWidgets.QApplication(sys.argv)
    window = GUI(framework_title)
    window.setCamera(cam, t_cam)
    window.setNetwork(network, t_network)
    window.show()


    # Threading GUI
    t_gui = ThreadGUI(window)
    t_gui.start()


    print("")
    print("Framework used: %s" %(framework_title))
    print("Requested timers:")
    print("    Camera: %d ms" % (t_cam.t_cycle))
    print("    GUI: %d ms" % (t_gui.t_cycle))
    print("    Network: %d ms" % (t_network.t_cycle))
    print("")

    sys.exit(app.exec_())
