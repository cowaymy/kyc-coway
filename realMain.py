
import os
os.environ["KMP_DUPLICATE_LIB_OK"]="TRUE"

import math
import time
import cv2 
import cvzone

from ultralytics.utils.plotting import save_one_box

from PIL import Image
from util import checkBlink


from ultralytics import YOLO

import matplotlib.pyplot as plt

confidence = 0.8

cap = cv2.VideoCapture(0)  # For Webcam
cap.set(3, 480)
cap.set(4, 480)

# cap = cv2.VideoCapture("../Videos/motorbikes.mp4")  # For Video

model = YOLO("./models/coway_real_face_best_v0.1.pt")
classNames = ["fake", "real"]


prev_frame_time = 0
new_frame_time = 0

blinkCounter=0
while_flag = True
beferImage =[]


while while_flag:
    # if cap.get(cv2.CAP_PROP_POS_FRAMES) == cap.get(cv2.CAP_PROP_FRAME_COUNT):
    #     cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
    new_frame_time = time.time()
    success, img = cap.read()
    results = model(img, stream=True, verbose=False)


    for r in results:
        boxes = r.boxes
        for box in boxes:
            # Bounding Box
            x1, y1, x2, y2 = box.xyxy[0]
            x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
            # cv2.rectangle(img,(x1,y1),(x2,y2),(255,0,255),3)
            w, h = x2 - x1, y2 - y1
            # Confidence
            conf = math.ceil((box.conf[0] * 100)) / 100
            # Class Name
            cls = int(box.cls[0])
            if conf > confidence:
                
                if classNames[cls] == 'real':
                    
                    bcolor = (0, 255, 0)

                    ############### 눈깜빡임 감지  ##################
                    if checkBlink(img) :
                        blinkCounter += 1
                        if blinkCounter >= 5:
                            while_flag =False
                            break

                    else :
                        beferImage.append(save_one_box(box.xyxy, r.orig_img.copy(), save=False))
                    ############### 눈깜빡임 감지  ##################



                    # if faces:
                    #     face = faces[0]

                    #     #for id in idList:
                    #         #cv2.circle(img, face[id], 5,color, cv2.FILLED)

                    #     leftUp = face[159]
                    #     leftDown = face[23]
                    #     leftLeft = face[130]
                    #     leftRight = face[243]
                    #     lenghtVer, _ = detector.findDistance(leftUp, leftDown)
                    #     lenghtHor, _ = detector.findDistance(leftLeft, leftRight)

                    #     #cv2.line(img, leftUp, leftDown, (0, 200, 0), 3)
                    #     #cv2.line(img, leftLeft, leftRight, (0, 200, 0), 3)

                    #     ratio = int((lenghtVer / lenghtHor) * 100)
                    #     ratioList.append(ratio)
                    #     if len(ratioList) > 3:
                    #         ratioList.pop(0)
                    #     ratioAvg = sum(ratioList) / len(ratioList)

                    #     if ratioAvg < 35 and counter == 0:
                    #         blinkCounter += 1
                    #         color = (0,200,0)
                    #         counter = 1
                    #     else:
                    #         beferImage.append(save_one_box(box.xyxy, r.orig_img.copy(), save=False))

                    #     if counter != 0:
                    #         counter += 1
                    #         if counter > 10:
                    #             counter = 0
                    #             color = (255,0, 255)

                    #     cvzone.putTextRect(img, f'Blink Count: {blinkCounter}', (50, 100),
                    #                         colorR=color)

                    #     imgPlot = plotY.update(ratioAvg, color)
                    #     img = cv2.resize(img, (640, 480))
                    #     imgStack = cvzone.stackImages([img, imgPlot], 2, 1)
                    # else:
                    #         img = cv2.resize(img, (640, 480))
                    #         imgStack = cvzone.stackImages([img, img], 2, 1)

                    # if blinkCounter == 2:
                    #     while_flag =False
                    #     break                    
                else:
                    bcolor = (0, 0, 255)
                    blinkCounter=0

                cvzone.cornerRect(img, (x1, y1, w, h),colorC=bcolor,colorR=bcolor)
                # cvzone.putTextRect(img, f'{classNames[cls].upper()} {int(conf*100)}%',
                #                    (max(0, x1), max(35, y1)), scale=2, thickness=4,colorR=bcolor,
                #                    colorB=bcolor)
                

    fps = 1 / (new_frame_time - prev_frame_time) 
    prev_frame_time = new_frame_time


    cv2.imshow("Image", img)
    if cv2.waitKey(1) == ord('q'):
        break


befImgLen =len(beferImage)

perValue = befImgLen * 70 // 100
Image.fromarray(beferImage[perValue]).save("./datasets/real/capture/images/real/realFace.jpg")
cv2.imshow("Image", img)


fig,axs= plt.subplots(1,2 , figsize=(15,5))
axs[0].imshow(img)
axs[1].imshow(plt.imread(r'datasets\real\capture\images\real\realFace.jpg'))

plt.show()
#cv2.destroyAllWindows()    
