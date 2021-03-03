
from ssd_predict import detector
from PIL import Image
import cv2
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt

###########################
NETWORK = "" # Choose an option in `shufflenetv1` or `shufflenetv2`
WEIGHT_FILE = "" # The network weigth file path.
###########################

tf.config.set_visible_devices([], 'GPU')
visible_devices = tf.config.get_visible_devices()
for device in visible_devices:
    assert device.device_type != "GPU"

image = Image.open("./image/06694.jpg")
predicts = []

det = detector(weight_path=WEIGHT_FILE, network=NETWORK)
image = det.detect_image(image)
plt.imshow(image)
plt.show()