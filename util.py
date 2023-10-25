import string
import easyocr
import numpy as np
import cv2, dlib
import datetime
import uuid
import face_recognition
import matplotlib.pyplot as plt
import pandas as pd




from PIL import Image
from pathlib import Path  
from imutils import face_utils
from keras.models import load_model

from ultralytics import YOLO
from ultralytics.utils.plotting import save_one_box

# for 
detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor('shape_predictor_68_face_landmarks.dat')
model = load_model('models/2018_12_17_22_58_35.h5')
model.summary()





# load models
coco_model = YOLO('yolov8n.pt')
coway_model = YOLO('./models/coway_nric_bast_v0.1.pt')



# fro ocr 
reader = easyocr.Reader(['en'], gpu=True)


def df_to_json(results):
    print(results.to_json(orient="records"))

   
def df_to_cvs(results):
    
    x = datetime.datetime.today().strftime("%Y%m%d%H%M%S")
    filepath = Path(f'folder/subfolder/{x}_out.csv') 
    filepath.parent.mkdir(parents=True, exist_ok=True)
    results.to_csv(filepath)  
  


def read_license_plate(license_plate_crop):
    """
    Read the license plate text from the given cropped image.

    Args:
        license_plate_crop (PIL.Image.Image): Cropped image containing the license plate.

    Returns:
        tuple: Tuple containing the formatted license plate text and its confidence score.
    """

    try:
        detections = reader.readtext(license_plate_crop)
        
        lineText =''
        for detection in detections:
                bbox,text,score = detection
                lineText += text.upper()

        return lineText , score    

    except :
        return  None , None

        #     return format_license(text), score

        # if license_complies_format(text):

    #return None, none


IMG_SIZE = (34, 26)

def crop_eye(img, eye_points):

  x1, y1 = np.amin(eye_points, axis=0)
  x2, y2 = np.amax(eye_points, axis=0)
  cx, cy = (x1 + x2) / 2, (y1 + y2) / 2

  w = (x2 - x1) * 1.2
  h = w * IMG_SIZE[1] / IMG_SIZE[0]

  margin_x, margin_y = w / 2, h / 2

  min_x, min_y = int(cx - margin_x), int(cy - margin_y)
  max_x, max_y = int(cx + margin_x), int(cy + margin_y)

  eye_rect = np.rint([min_x, min_y, max_x, max_y]).astype(int)

  eye_img = img[eye_rect[1]:eye_rect[3], eye_rect[0]:eye_rect[2]]

  return eye_img, eye_rect


def checkBlink(img_ori):
    
    blink_Falg =False

    img_ori = cv2.resize(img_ori, dsize=(0, 0), fx=0.5, fy=0.5)

    img = img_ori.copy()
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    faces = detector(gray)

    for face in faces:
        shapes = predictor(gray, face)
        shapes = face_utils.shape_to_np(shapes)

        eye_img_l, eye_rect_l = crop_eye(gray, eye_points=shapes[36:42])
        eye_img_r, eye_rect_r = crop_eye(gray, eye_points=shapes[42:48])

        eye_img_l = cv2.resize(eye_img_l, dsize=IMG_SIZE)
        eye_img_r = cv2.resize(eye_img_r, dsize=IMG_SIZE)
        eye_img_r = cv2.flip(eye_img_r, flipCode=1)

    
        eye_input_l = eye_img_l.copy().reshape((1, IMG_SIZE[1], IMG_SIZE[0], 1)).astype(np.float32) / 255.
        eye_input_r = eye_img_r.copy().reshape((1, IMG_SIZE[1], IMG_SIZE[0], 1)).astype(np.float32) / 255.

        pred_l = model.predict(eye_input_l)
        pred_r = model.predict(eye_input_r)



        # visualize
        state_l = 'O %.1f' if pred_l > 0.1 else '- %.1f'
        state_r = 'O %.1f' if pred_r > 0.1 else '- %.1f'

        state_l = state_l % pred_l
        state_r = state_r % pred_r

        if pred_r[0][0] <= 0.15 and   pred_l[0][0] <= 0.15:
            blink_Falg =True
            break
        

    return blink_Falg



df = pd.DataFrame( columns =['tran_id','idx' ,'type','value', 'bbox_conf','value_conf','cropFaceImg'])

def  OCR(inputImg) :

    rIdx =0
    res = coway_model(f'{inputImg}')

    for r in res:
        trin_id  =uuid.uuid4()
        results = {}
        
        arr = r.plot()
        image = Image.fromarray(arr[..., ::-1])
        r.save_crop('datasets/yolo/images/crop' ,file_name=f'{trin_id}.jpg')
        cropImagName ="datasets/yolo/images/crop/IMAGE/"f'{trin_id}'".jpg"

	
        for  indx, d in  enumerate(r.boxes):
            if r.names[int(d.cls)] != 'IMAGE':
                image = save_one_box(d.xyxy, r.orig_img.copy(), BGR=True, save=False)
                license_plate_text, license_plate_text_score = read_license_plate(image)
            
                df.loc[rIdx]={'tran_id': str(trin_id),
                'idx':indx ,
                'type':r.names[int(d.cls)] ,
                'value':license_plate_text ,
                'bbox_conf' :float(d.conf), 
                'value_conf' :license_plate_text_score,
                'cropFaceImg':f'{str(trin_id)}.jpg'}

                rIdx = rIdx+1
                
            # else:
            #     image = save_one_box(d.xyxy, r.orig_img.copy(), BGR=True, save=False)
            #     image = Image.fromarray(image)

            #     face = face_recognition.face_encodings(np.array(image))[0]
            #     realface = face_recognition.face_encodings(face_recognition.load_image_file(r".\datasets\real\capture\images\real\realFace.jpg"))[0]
            #     results = face_recognition.compare_faces([face], realface, tolerance=0.45)
            #     print(results)

            #     fig,axs= plt.subplots(1,2 , figsize=(15,5))
            #     axs[0].imshow(plt.imread(cropImagName))
            #     axs[1].imshow(plt.imread(r'datasets\real\capture\images\real\realFace.jpg'))

            #     fig.suptitle(f" Verifie :: { results }")
            #     plt.show()


    return df


def call_face_detection(scanimage , realFaceImg):

        print ('scanimage===>' , scanimage)
        print ('realFaceImg===>' , realFaceImg)
        
        #face = face_recognition.face_encodings(np.array(image))[0]
        m1 = face_recognition.load_image_file(scanimage)
        face = face_recognition.face_encodings(m1)[0]
        
        realface = face_recognition.face_encodings(face_recognition.load_image_file(realFaceImg))[0]
        results = face_recognition.compare_faces([face], realface, tolerance=0.45)
        print(results)

        # fig,axs= plt.subplots(1,2 , figsize=(15,5))
        # axs[0].imshow(plt.imread(scanimage))
        # axs[1].imshow(plt.imread(realFaceImg))
    
        # fig.suptitle(f" Verifie :: { results }")
        # plt.show()

        return results

        
