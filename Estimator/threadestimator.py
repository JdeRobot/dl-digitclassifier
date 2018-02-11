#!/usr/bin/env python

"""
threadestimator.py: Thread for Estimator class.

Based on https://github.com/RoboticsURJC-students/2016-tfg-david-pascual
"""
__author__ = "David Pascual Hernandez"
__date__ = "2017/11/16"

import threading
import time
from datetime import datetime

t_cycle = 50  # ms


class ThreadEstimator(threading.Thread):
    def __init__(self, estimator):
        """
        Threading class for estimator.
        @param estimator: Estimator object
        """
        self.estimator = estimator
        threading.Thread.__init__(self)

    def run(self):
        """ Updates the thread. """
        while True:
            start_time = datetime.now()
            self.estimator.update()
            end_time = datetime.now()

            dt = end_time - start_time
            dtms = ((dt.days * 24 * 60 * 60 + dt.seconds) * 1000
                    + dt.microseconds / 1000.0)

            if dtms < t_cycle:
                time.sleep((t_cycle - dtms) / 1000.0)
