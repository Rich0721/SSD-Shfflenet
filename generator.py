from random import shuffle
import numpy as np
import cv2
from PIL import Image
from tensorflow.keras.applications.imagenet_utils import preprocess_input


def rand(a=0, b=1):
    return np.random.rand()*(b-a) + a


class Generator:
    
    def __init__(self, bbox_util, batch_size, train_lines, val_lines, image_size, num_classes) -> None:
        
        self.bbox_util = bbox_util
        self.batch_size = batch_size
        self.train_lines = train_lines
        self.val_lines = val_lines
        self.train_batch = len(train_lines)
        self.val_batch = len(val_lines)
        self.image_size = image_size
        self.num_classes = num_classes
    
    def data_argumation(self, annotation_line, input_shape, jitter=.3, hue=.1, sat=1.5, val=1.5, random=True):

        line = annotation_line.split()
        image = Image.open(line[0])
        iw, ih = image.size
        h, w = input_shape
        box = np.array([np.array(list(map(int, box.split(",")))) for box in line[1:]])

        if not random:

            scale = min(w/iw, h/ih)
            nw = int(iw * scale)
            nh = int(ih * scale)
            dx = (w - nw) // 2
            dy = (h - nh) // 2

            image = image.resize((nw, nh), Image.BICUBIC)
            new_image = Image.new('RGB', (w, h), (128, 128, 128))
            new_image.paste(image, (dx, dy))
            image_data = np.array(new_image, np.float32)

            # correct boxes
            box_data = np.zeros((len(box), 5))
            if len(box) > 0:
                np.random.shuffle(box)
                box[:, [0, 2]] = box[:, [0, 2]] * nw / iw + dx
                box[:, [1, 3]] = box[:, [1, 3]] * nh / ih + dy
                box[:, 0:2][box[:, 0:2] < 0] = 0
                box[:, 2][box[:, 2] > w] = w
                box[:, 3][box[:, 3] > h] = h
                box_w = box[:, 2] - box[:, 0]
                box_h = box[:, 3] - box[:, 1]
                box = box[np.logical_and(box_w > 1, box_h > 1)]
                box_data[:len(box)] = box
            return image_data, box_data

         # resize image
        new_ar = w/h * rand(1-jitter, 1+jitter) / rand(1-jitter, 1+jitter)
        scale = rand(0.5, 1.5)
        if new_ar < 1:
            nh = int(scale * h)
            nw = int(nh * new_ar)
        else:
            nw = int(scale * w)
            nh = int(nw / new_ar)
        image = image.resize((nw, nh), Image.BICUBIC)

        dx = int(rand(0, w - nw))
        dy = int(rand(0, h - nh))
        new_image = Image.new('RGB', (w, h), (128, 128, 128))
        new_image.paste(image, (dx, dy))
        image = new_image

        flip = rand() < 0.5
        if flip:
            image = image.transpose(Image.FLIP_LEFT_RIGHT)
        
        #  Gaussian Noise
        if rand() < 0.5:
            gauss = np.random.normal(0, 0.1**0.5, (w, h, 3))
            gauss = gauss.reshape(w, h, 3)
            image = image + gauss
            
        # distort image
        hue = rand(-hue, hue)
        sat = rand(1, sat) if rand() < 0.5 else 1 / rand(1, sat)
        val = rand(1, val) if rand() < 0.5 else 1 / rand(1, val)
        x = cv2.cvtColor(np.array(image, np.float32) / 255, cv2.COLOR_RGB2HSV)
        x[..., 0] += hue*360
        x[..., 0][x[..., 0] > 1] -= 1
        x[..., 0][x[..., 0] < 0] += 1
        x[..., 1] *= sat
        x[..., 2] *= val
        x[x[:, :, 0] > 360, 0] = 360
        x[:, :, 1:][x[:, :, 1:] > 1] = 1
        x[x < 0] = 0
        image_data = cv2.cvtColor(x, cv2.COLOR_HSV2RGB) * 255

        # correct box
        box_data = np.zeros((len(box), 5))
        if len(box) > 0:
            np.random.shuffle(box)
            box[:, [0, 2]] = box[:, [0, 2]] * nw / iw + dx
            box[:, [1, 3]] = box[:, [1, 3]] * nh / ih + dy
            if flip: box[:,[0, 2]] = w - box[:, [2, 0]]
            box[:, 0:2][box[:, 0:2] < 0] = 0
            box[:, 2][box[:, 2] > w] = w
            box[:, 3][box[:, 3] > h] = h
            box_w = box[:, 2] - box[:, 0]
            box_h = box[:, 3] - box[:, 1]
            box = box[np.logical_and(box_w > 1, box_h > 1)]
            box_data[:len(box)] = box
        
        return image_data, box_data
    
    def generator(self, train=True):

        while True:
            if train:
                shuffle(self.train_lines)
                lines = self.train_lines
            else:
                shuffle(self.val_lines)
                lines = self.val_lines
            
            inputs = []
            targets = []
            for annotation_line in lines:
                if train:
                    img, y = self.data_argumation(annotation_line, self.image_size[0:2])
                else:
                    img, y = self.data_argumation(annotation_line, self.image_size[0:2], random=True)
                
                if len(y) != 0:
                    boxes = np.array(y[:, :4], dtype=np.float32)
                    boxes[:,0] = boxes[:,0]/self.image_size[1]
                    boxes[:,1] = boxes[:,1]/self.image_size[0]
                    boxes[:,2] = boxes[:,2]/self.image_size[1]
                    boxes[:,3] = boxes[:,3]/self.image_size[0]
                    one_hot_label = np.eye(self.num_classes)[np.array(y[:,4],np.int32)]
                    if ((boxes[:,3]-boxes[:,1])<=0).any() and ((boxes[:,2]-boxes[:,0])<=0).any():
                        continue
                    y = np.concatenate([boxes,one_hot_label],axis=-1)
                
                y = self.bbox_util.assign_boxes(y)
                inputs.append(img)
                targets.append(y)
                if len(targets) == self.batch_size:
                    tmp_inp = np.array(inputs)
                    tmp_targets = np.array(targets)
                    inputs = []
                    targets = []
                    yield preprocess_input(tmp_inp), tmp_targets

    def load_image_and_annotations(self, i):

        line = self.val_lines[i].split()
        image = cv2.imread(line[0])
        image = cv2.resize(image, (self.image_size[1], self.image_size[1]))
        image = image[:, :, ::-1] / 255
        boxes = np.array([np.array(list(map(int, box.split(",")))) for box in line[1:]])
        return image, boxes