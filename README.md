# Digit Classifier
<code>digitclassifier</code> is a JdeRobot component which captures live video and classifies the digits found in every frame with a convolutional neural network. Implementations in both Keras and Tensorflow are available.
## Usage
In order to test it with Python 2.7 you must install: 
* JdeRobot ([installation guide](http://jderobot.org/Installation))
* OpenCV 3 (it will be automatically installed with JdeRobot)

Aditionally, you will need a few Python packages, generally installable via <code>python-pip</code>. We have prepared a <code>requirements.txt</code> file, which will automatically install all these dependencies by running:
<code>pip2 installl -r requirements.txt </code>


If you want to launch <code>digitclassifier</code>, open a terminal and run:
<pre>
cameraserver cameraserver.cfg
</pre>
This command will start <code>cameraserver</code> driver, which will serve 
video from the webcam. In another terminal run:
<pre>
python digitclassifier.py digitclassifier.yml
</pre>
That command should launch the component and you should see something like this:
![Alt Text](https://media.giphy.com/media/xT0xevE4RgzA4CTEju/giphy.gif)

## YAML config. file
<code>digitclassifier.yml</code> file contains fields to choose which framework
to use during live digit classification (<code>Framework</code>), as well
as the path to the corresponding model (<code>Model</code>). For example, if 
you want to test Keras model, your config. file should look something like 
this: 
<pre>
...
  Framework: "Keras"  # Currently supported: "Keras" or "Tensorflow"
  Model: "Estimator/Keras/Model/net.h5"  # path to model
...
</pre>

And if you want to test TensorFlow model:
<pre>
...
  Framework: "TensorFlow"  # Currently supported: "Keras" or "Tensorflow"
  Model: "Estimator/TensorFlow/mnist-model/"  # path to model
...
</pre>


## Datasets
If you want to train a new model with the augmented MNIST datasets we used for
training, here they are:
* [HDF5 datasets](https://mega.nz/#!hV12GapC!3eGRv0Ty8VRoJxhnbrG_4e21QUnPNjraTnqUJog7PxU)
* [LMDB datasets](https://mega.nz/#!NBkBTSRI!TPfLk4nHY5WjconmhbI9jV_yZLvnImDzextQSBcA6Wk)


## More info
About Keras implementation: [http://jderobot.org/Dpascual-tfg] \
About TensorFlow implementation: [http://jderobot.org/Naxvm-tfg]

