#!/usr/bin/env python

"""
estimator.py: Estimator class.
"""
__author__ = "David Pascual Hernandez"
__date__ = "2017/11/16"

class Estimator:
    """
    Estimator class makes a prediction given an input image.
    """
    def __init__(self, gui, cam, data):
        """
        Estimator object constructor.
        @param gui: GUI object
        @param cam: Camera object
        @param data: parsed YAML config. file
        """
        self.gui = gui
        self.cam = cam

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

    def update(self):
        """
        Update estimator.
        """
        _, im = self.cam.get_image()
        digit = self.estimate(im)

        self.gui.pred = digit
