
import os
os.environ["KMP_DUPLICATE_LIB_OK"]="TRUE"

from ultralytics import YOLO

model =YOLO('yolov8n.pt')


def main() :
    model.train(data ='datasets/splitData/dataOffline.yaml' , epochs=3)



if __name__ == '__main__':
    main()
