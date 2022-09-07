import mediapipe as mp
import cv2
import numpy as np
import time

mp_draw = mp.solutions.drawing_utils
mp_face_mesh = mp.solutions.face_mesh
face_mesh = mp_face_mesh.FaceMesh(max_num_faces=3,
    min_detection_confidence=0.2)
circleDrawingSpec = mp_draw.DrawingSpec(thickness=0, circle_radius=0, color=(255,255,255))
lineDrawingSpec = mp_draw.DrawingSpec(thickness=2, color=(0,255,0))
mp_face = mp.solutions.face_detection
face = mp_face.FaceDetection()
s = time.time()
cap = cv2.VideoCapture(0)
while cap.isOpened():
    ret, img = cap.read()
    rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    results = face_mesh.process(rgb)
    h, w, _ = img.shape
    # faces = face.process(rgb).detections
    try:
        # print("Number of People: ", len(faces))
        landmarks =results.multi_face_landmarks 
        for lm in landmarks:
            mp_draw.draw_landmarks(img, lm, mp_face_mesh.FACEMESH_CONTOURS, circleDrawingSpec, lineDrawingSpec)
    except:
        continue
    e = time.time()
    fps = 1/(e-s)
    s = e
    cv2.putText(img, f'FPS : {int(fps)}', (20, 100), cv2.FONT_HERSHEY_DUPLEX, 1, (255, 0, 0), 2)
    cv2.imshow("Image", img)
    if(cv2.waitKey(1) == ord('q')):
        break
cap.release()
cv2.destroyAllWindows()
