import numpy as np
from config import config

IOU_THRESHOLD = 0.75

def compute_ap(recall, precision):
    
    mrec = np.concatenate(([0.], recall, [1.]))
    mpre = np.concatenate(([0.], precision, [0.]))
    
    for i in range(mpre.size -1, 0, -1):
        mpre[i-1] = np.maximum(mpre[i-1], mpre[i])
    
    i = np.where(mrec[1:] != mrec[:-1])[0]
    ap = np.sum((mrec[i+1] - mrec[i]) * mpre[i+1])
    return ap
    
def compute_iou(b1, b2):

    inter_xmin = max(b1[0], b2[0])
    inter_ymin = max(b1[1], b2[1])
    inter_xmax = min(b1[2], b2[2])
    inter_ymax = min(b1[3], b2[3])
    
    inter_w = max(0, inter_xmax - inter_xmin)
    inter_h = max(0, inter_ymax - inter_ymin)

    inter_area = inter_w * inter_h
    b1_area =  (b1[2] - b1[0]) * (b1[3] - b1[1])
    b2_area =  (b2[2] - b2[0]) * (b2[3] - b2[1])
    return max(0,inter_area / (b1_area + b2_area - inter_area))  


def compute_average_precision(init_file, result_file):

    with open(init_file) as f:
        init_lines = f.readlines()

    with open(result_file) as f:
        result_lines = f.readlines()
    TP = np.zeros(len(config.CLASSES), dtype=np.int32)
    FP = np.zeros(len(config.CLASSES), dtype=np.int32)
    NUM_ANNOTATION = np.zeros(len(config.CLASSES), dtype=np.int32) 
    #print(all_detections)

    for index in range(len(result_lines)):

        original = init_lines[index].split()
        result = result_lines[index].split()
        
        original_bboxes = np.array([np.array(list(map(int, box.split(",")))) for box in original[1:]])
        result_bboxes = np.array([np.array(list(map(int, box.split(",")))) for box in result[1:]])


        for original_box in original_bboxes:

            NUM_ANNOTATION[original_box[-1]]+=1
            classes = -1
            for result_bbox in result_bboxes:

                iou = compute_iou(original_box, result_bbox)
                if iou > IOU_THRESHOLD:
                    classes = result_bbox[-1]
                    break
        
            if classes == original_box[-1]:
                TP[classes]+=1
            else:
                FP[classes]+=1

    pre = TP/(TP+FP)
    recall = TP / NUM_ANNOTATION
    
    print("AP:{:.2f}".format(compute_ap(recall, pre)))


def union_commtity(init_file, result_file):
    
    all = 0
    correct = 0
    with open(init_file) as f:
        init_lines = f.readlines()

    with open(result_file) as f:
        result_lines = f.readlines()
    
    for index in range(len(result_lines)):

        original = init_lines[index].split()
        result = result_lines[index].split()
        
        original_bboxes = np.array([np.array(list(map(int, box.split(",")))) for box in original[1:]])
        result_bboxes = np.array([np.array(list(map(int, box.split(",")))) for box in result[1:]])

        init_classes = []
        predict_classes = []
        for bboxes in original_bboxes:
            init_classes.append(bboxes[4])
        for bboxes in result_bboxes:
            predict_classes.append(bboxes[4])
        
        init_classes = set(init_classes)
        predict_classes = set(predict_classes)
        intersection = init_classes.intersection(predict_classes)
        union = init_classes.union(predict_classes)
        if len(union) == len(intersection):
            correct+=1
        all+=1

    print("correct:{}, all:{}, av:{:.2f}".format(correct, all, correct/all))

if __name__ == "__main__":
    compute_average_precision(init_file="test.txt", result_file="vgg_result.txt")
    union_commtity(init_file="test.txt", result_file="vgg_result.txt")