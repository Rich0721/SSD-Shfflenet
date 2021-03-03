
class config:

    EPOCHS = 250
    BATCH_SIZE = 8
    IMAGE_SIZE_300 = (300, 300, 3)
    ANCHORS_SIZE_300 = [30, 60, 111, 162, 213, 264, 315] # VOC SSD300
    VARIANCES = [0.1, 0.1, 0.2, 0.2]

    CLASSES = ['1402200300101', '1402300300101', '1402310200101', '1402312700101', '1402312900101', 
                '1402324800101', '1422001900101', '1422111300101', '1422204600101', '1422206800101', 
                '1422300200101', '1422300300101', '1422301800101', '1422302000101', '1422305100101', 
                '1422305900101', '1422308000101', '1422329600101', '1422503600101', '1422504400101', 
                '1422505200101', '1422505600101', '1422593400101', '1422594600101', '1423003100101', 
                '1423014700101', '1423100700101', '1423103600101', '1423206800101', '1423207800101', '1423301600101']
    
    DATASET = ["../datasets/test_network/"]
    VOC_TEXT_FILE = ["../datasets/test_network/train.txt", "../datasets/test_network/val.txt", "../datasets/test_network/test.txt"] # Dataset train, val, test images information.
    VOC_TRAIN_FILE = ["./train.txt", "./val.txt", "./test.txt"] # Produce train text file.

    CONFIDENCE = 0.5
    NMS_IOU = 0.45
    MODEL_FOLDER = "./ssd_vgg/" # Stroage weigth folder
    FILE_NAME = "ssd_vgg" # Storage weigth weight
    