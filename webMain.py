
import os
os.environ["KMP_DUPLICATE_LIB_OK"]="TRUE"

import streamlit  as st
import base64 
import json
import pandas as pd
import cv2
import math
import time
import cvzone

from ultralytics.utils.plotting import save_one_box
from ultralytics import YOLO
from streamlit_extras.stylable_container import stylable_container
from streamlit_extras.row import row 




from PIL import Image
from io import BytesIO
from datetime import datetime
from util import OCR ,checkBlink ,call_face_detection
import matplotlib.pyplot as plt



st.set_page_config(layout="wide", page_title="welcome to coway ")


row1 = row([2, 4, 1], vertical_align="bottom")
row1.empty()
row1.write("## smart e-keyin Prototype")
row1.empty()

st.sidebar.markdown(
    '<div style="margin: 0.75em 0;"><a href="#" target="_blank"><img src="https://www.coway.com.my/img/coway-malaysia-logo.png" alt="coway" height="41" width="174"></a></div>',
    unsafe_allow_html=True,
)


st.sidebar.markdown("---")

st.sidebar.write("##  Upload  your NRIC :gear:")

MAX_FILE_SIZE = 5 * 1024 * 1024  # 5MB

################## OCR Result ##################
ocr_nric   = ''
ocr_name   = ''
ocr_gender = ''
ocr_img = ''
cropFaceImg =''



################ face_detection #################
confidence = 0.8

# cap = cv2.VideoCapture("../Videos/motorbikes.mp4")  # For Video

model = YOLO("./models/coway_real_face_best_v0.1.pt")
classNames = ["fake", "real"]



prev_frame_time = 0
new_frame_time = 0

blinkCounter=0
while_flag = True
beferImage =[]



def getOCRdata(img):

    data = OCR(img)
    df = pd.DataFrame(data)

    for index, row in df.iterrows():
        if str(row['type']) == 'IC' :
            ocr_nric =row['value']
            imgpath =row['cropFaceImg']
            ocr_img =f"datasets/yolo/images/crop/IMAGE/{imgpath}" 
            cropFaceImg =imgpath

            

        if str(row['type']) == 'NAME' :
            ocr_name =row['value'] 

        if str(row['type']) == 'GENDER' :
            ocr_gender =row['value'] 


    #st.table(data)    

    return (ocr_img, ocr_nric, ocr_name, ocr_gender, cropFaceImg)
    st.table(data)




# 파일 업로드 함수
# 디렉토리 이름, 파일을 주면 해당 디렉토리에 파일을 저장해주는 함수
def save_uploaded_file(directory, file):
    # 1. 저장할 디렉토리(폴더) 있는지 확인
    #   없다면 디렉토리를 먼저 만든다.
    if not os.path.exists(directory):
        os.makedirs(directory)
    
    # 2. 디렉토리가 있으니, 파일 저장
    with open(os.path.join(directory, file.name), 'wb') as f:
        f.write(file.getbuffer())

        return getOCRdata(f'image/upload/{file.name}')
    return st.success('파일 업로드 성공!')

def endLayout():
    row2 = row([2, 4, 1], vertical_align="bottom")

    row2.empty()
    row2.empty()
    row2.st.markdown(
        '<div style="margin: 0.75em 0;"><a href="#" target="_blank"><img src="https://www.coway.com.my/img/coway-malaysia-logo.png" alt="coway" height="41" width="174"></a></div>',
        unsafe_allow_html=True,
    )






tempRealImgPath ='./datasets/real/capture/images/real/temp.png'
take_face=False

# def start_face_detection(scanImg):

#     confidence = 0.8


#     blinkCounter=0
#     while_flag = True
#     beferImage =[]

        
#     cap = cv2.VideoCapture(0)
#     stframe = st.empty()

        
#     while while_flag: 
#         ret, img = cap.read()
#         results = model(img, stream=True, verbose=False)

#         # if frame is read correctly ret is True
#         if not ret:
#             print("Can't receive frame (stream end?). Exiting ...")
#             break


#         for r in results:
#             boxes = r.boxes
#             for box in boxes:
#                 # Bounding Box
#                 x1, y1, x2, y2 = box.xyxy[0]
#                 x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
#                 # cv2.rectangle(img,(x1,y1),(x2,y2),(255,0,255),3)
#                 w, h = x2 - x1, y2 - y1
#                 # Confidence
#                 conf = math.ceil((box.conf[0] * 100)) / 100
#                 # Class Name
#                 cls = int(box.cls[0])

#                 if conf > confidence:
#                     if classNames[cls] == 'real':
#                         bcolor = (0, 255, 0)

#                         print('================================>' ,blinkCounter)
#                         ############### 눈깜빡임 감지  ##################
#                         if checkBlink(img) :
#                             blinkCounter += 1
#                             if blinkCounter >= 5:
#                                 while_flag =False
#                                 break

#                         else :
#                             beferImage.append(save_one_box(box.xyxy, r.orig_img.copy(), save=False))
#                         ############### 눈깜빡임 감지  ##################
#                     else:
#                         bcolor = (0, 0, 255)
#                         blinkCounter=0

#                     cvzone.cornerRect(img, (x1, y1, w, h),colorC=bcolor,colorR=bcolor)
#                     # cvzone.putTextRect(img, f'{classNames[cls].upper()} {int(conf*100)}%',
#                     #                    (max(0, x1), max(35, y1)), scale=2, thickness=4,colorR=bcolor,
#                     #                    colorB=bcolor)
    
#         gray = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
#         stframe.image(gray)

#     befImgLen =len(beferImage)

#     perValue = befImgLen * 70 // 100
#     Image.fromarray(beferImage[perValue]).save("./datasets/real/capture/images/real/realFace.jpg")
#     tempRealImgPath ='./datasets/real/capture/images/real/realFace.jpg'


#     take_face =True

#     tcol1, tcol2 = st.columns(2)

#     if take_face :
#         with tcol1:
#             st.image(f'{scanImg}' ,width=200)
#         with tcol2:
#             st.image(f'{tempRealImgPath}',width=200)

#     result = call_face_detection(scanImg,tempRealImgPath)
#     st.header(f'RESULT::: {result}')




row2 = row([2, 4, 1], vertical_align="bottom")

row2.empty()
row2.empty()
row2.write(
    '<div style="margin: 0.75em 0;"><a href="#" target="_blank"><img src="https://www.coway.com.my/img/coway-malaysia-logo.png" alt="coway" height="41" width="174"></a></div>',
    unsafe_allow_html=True,
)



st.subheader('1st OCR Result with a divider', divider='rainbow')     
col1, col2 = st.columns(2)
img_file = st.sidebar.file_uploader("Upload an image", type=["png", "jpg", "jpeg"],accept_multiple_files=False)





if img_file is not None: # 파일이 없는 경우는 실행 하지 않음

    # 유저가 올린 파일을,
    # 서버에서 처리하기 위해서(유니크하게) 
    # 파일명을 현재 시간 조합으로 만든다. 
    current_time = datetime.now()
    # print(current_time)
    # print(current_time.isoformat().replace(':', "_") + '.jpg') #문자열로 만들어 달라
    # # 파일 명에 특정 특수문자가 들어가면 만들수 없다.
    filename = current_time.isoformat().replace(':', "_") + '.jpg'
    img_file.name = filename

    return_data = save_uploaded_file('image/upload', img_file)


    with col1:
        st.header(" UPLOAD(업로드) ")
        st.image(f'image/upload/{img_file.name}')


    with col2:
        st.header(" RESULT(결과)")
        st.divider()

        st.text_input('NRIC', return_data[1])
        st.text_input('NAME', return_data[2])
        st.text_input('GENDER',return_data[3])
        st.divider()
        # st.image(return_data[0], width=200)   
        scanImg =str(return_data[0])
        cropFaceImg = return_data[4]
    
    st.subheader('2nd Scan Face Result with a divider', divider='rainbow')     
    
    ####### camera ######################
    camera_img = st.camera_input("Take a Your Face", disabled =False )
    ####### camera ######################
    if camera_img is not None:

        Image.open(camera_img).save(f'datasets/real/capture/images/real/{cropFaceImg}')
        tempRealImgPath =f'./datasets/real/capture/images/real/{cropFaceImg}'

        fresult=''    
        take_face =True
        results = model(f'./datasets/real/capture/images/real/{cropFaceImg}', stream=False, verbose=False)

        
        for r in results:
            boxes = r.boxes
                
            for box in boxes:
                # Bounding Box
                x1, y1, x2, y2 = box.xyxy[0]
                x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
                w, h = x2 - x1, y2 - y1
                # Confidence
                conf = math.ceil((box.conf[0] * 100)) / 100
                # Class Name
                cls = int(box.cls[0])
                
                fresult =classNames[cls] 

            
                #if conf > confidence:
                if classNames[cls] == 'real':
                    bcolor = (0, 255, 0)
                    fresult ='real'
                
                else:
                    bcolor = (0, 0, 255)
                    fresult ='fake'
                
                #else:
                # bcolor = (0, 0, 255)
                    

                #cv2.imshow("Image", save_one_box(box.xyxy, r.orig_img.copy(), save=False))
                cvimg =cvzone.cornerRect(r.orig_img.copy() ,(x1, y1, w, h),colorC=bcolor,colorR=bcolor)
                cvzone.putTextRect(cvimg, f'{classNames[cls].upper()} {int(conf*100)}%',(max(0, x1), max(35, y1)), scale=2, thickness=4,colorR=bcolor,)
                # 
                #                    

                break

        tcol1, tcol2 = st.columns(2)
        if take_face :
            with tcol1:
                st.image(f'{scanImg}' , width =300 )
            with tcol2:
                st.image(cv2.cvtColor(cvimg, cv2.COLOR_RGB2BGR) )

        result = call_face_detection(scanImg,tempRealImgPath)
        
        
        
        row3 = row([2, 4, 1], vertical_align="bottom")
        row3.empty()
        row3.header(f'RESULT::: {result} ')
        row3.empty()

        #st.header(f'RESULT::: {result} ')

        st.sidebar.markdown(
        '<div style="margin: 0.75em 0;"><a href="#" target="_blank"><img src="./app/static/coway-malaysia-logo.png" alt="coway" height="41" width="174"></a></div>',
        unsafe_allow_html=True,
    )



# st.sidebar.write("##  Open Camera  ")
# while_flag = st.sidebar.button("scan to face")



# if while_flag:
#     start_face_detection(scanImg)

