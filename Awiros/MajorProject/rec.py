import cv2
import mediapipe as mp
import time
from natsort import natsorted
import os
import numpy as np
import face_recognition as fr
from PIL import Image

def resize(img, size) :
    width = int(img.shape[1]*size)
    height = int(img.shape[0] * size)
    dimension = (width, height)
    return cv2.resize(img, dimension, interpolation= cv2.INTER_AREA)
def findEncodings(images):
    encodeList = []
    for img in images:
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        encode = fr.face_encodings(img)[0]
        encodeList.append(encode)
    return encodeList
def names_images(myList, path):
    names = []
    images = []
    myList = natsorted(myList)
    for imgNames in myList:
        curImg = cv2.imread(f"{path}/{imgNames}")
        images.append(curImg)
        names.append(os.path.splitext(imgNames)[0])
    return names, images

path = 'C:\\Users\\tejus\\Desktop\\Awiros\\MajorProject\\image'
myList = os.listdir(path)
names, images = names_images(myList, path)
t = len(names)
encodeListKnown = findEncodings(images)
mp_facedetector = mp.solutions.face_detection
mp_draw = mp.solutions.drawing_utils
# cap = cv2.VideoCapture("rtsp://admin:awicam6661@10.15.17.40:554/cam/realmonitor?channel=1&subtype=0")
cap = cv2.VideoCapture(0)
with mp_facedetector.FaceDetection(min_detection_confidence=0.9) as face_detection:
    i = 1
    while cap.isOpened():
        success, image = cap.read()
        start = time.time()
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        results = face_detection.process(image)
        image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
        if results.detections:
            for id, detection in enumerate(results.detections):
                # mp_draw.draw_detection(image, detection, mp_draw.DrawingSpec(color=(0, 0, 0), thickness = 0, circle_radius = 0), mp_draw.DrawingSpec(color=(0, 255, 0)))
                bBox = detection.location_data.relative_bounding_box
                h, w, c = image.shape
                xleft = int(bBox.xmin*w)
                xtop = int(bBox.ymin*h)
                xright = int(bBox.width*w + xleft)
                xbottom = int(bBox.height*h + xtop)
                boundBox = int(bBox.xmin * w), int(bBox.ymin * h), int(bBox.width * w), int(bBox.height * h)
                detFac = xleft, xtop, xright, xbottom
                face = Image.fromarray(image).crop(detFac)
                face = np.asarray(face)
                # face = resize(face, 0.25)
                encode = fr.face_encodings(face)
                if len(encode) == 0:
                    break
                result = fr.compare_faces(encodeListKnown, encode[0])
                if True not in result:# or len(result) == 0:
                    cv2.imwrite(f'C:/Users/tejus/Desktop/Awiros/MajorProject/image/{i}.jpg', face)
                    i+=1
                    names.append(str(i+t))
                    encodeListKnown.append(encode[0])
                    images.append(image)
                    result.append("True")
                faceDis = fr.face_distance(encodeListKnown, encode[0])
                print(faceDis)
                matchIndex = np.argmin(faceDis) 
                mp_draw.draw_detection(image, detection, mp_draw.DrawingSpec(color=(0, 0, 0), thickness = 0, circle_radius = 0), mp_draw.DrawingSpec(color=(0, 255, 0)))
                cv2.putText(image, f'{matchIndex+1}', (xleft + 6, xbottom + 20), cv2.FONT_HERSHEY_COMPLEX_SMALL, 1, (255, 0, 0), 2)
        end = time.time()
        totalTime = end - start
        if(totalTime):
            fps = 1 / totalTime
            cv2.putText(image, f'FPS: {int(fps)}', (20,70), cv2.FONT_HERSHEY_SIMPLEX, 1, (0,255,0), 2)
        # image = resize(image, 0.5)
        cv2.imshow('Face Detection', image)
        if cv2.waitKey(5) & 0xFF == 27:
            break
cap.release()
cv2.destroyAllWindows()