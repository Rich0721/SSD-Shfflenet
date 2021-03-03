from ssd_predict import detector
from PIL import Image
import cv2
import tensorflow as tf
import numpy as np

###########################
NETWORK = "" # Choose an option in `shufflenetv1` or `shufflenetv2`
WEIGHT_FILE = "" # The network weigth file path.
###########################

tf.config.set_visible_devices([], 'GPU')
visible_devices = tf.config.get_visible_devices()
for device in visible_devices:
    assert device.device_type != "GPU"

det = detector(weight_path=WEIGHT_FILE, network=NETWORK)

def use_video():
    
    frame_number = 0
    cap = cv2.VideoCapture(0)
    _ = True
    while True:
        _, frame = cap.read()
        
        if frame is None:
            break
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        frame = Image.fromarray(np.uint8(frame))
        frame = det.detect_image(frame)
        frame = cv2.cvtColor(np.asarray(frame), cv2.COLOR_RGB2BGR)
        frame_number+=1
        cv2.imshow("video", frame)
        cv2.waitKey(1)


if __name__ == "__main__":
    use_video()