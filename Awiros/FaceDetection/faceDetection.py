import cv2
import mediapipe as mp
import time

mp_facedetector = mp.solutions.face_detection
mp_draw = mp.solutions.drawing_utils

with mp_facedetector.FaceDetection(min_detection_confidence=0.9) as face_detection:
    cap = cv2.VideoCapture(0)
    while cap.isOpened():
        success, image = cap.read()
        start = time.time()
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        results = face_detection.process(image)
        image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
        if results.detections:
            for id, detection in enumerate(results.detections):
                mp_draw.draw_detection(image, detection, mp_draw.DrawingSpec(color=(0, 0, 0), thickness = 0, circle_radius = 0), mp_draw.DrawingSpec(color=(0, 255, 0)))
                bBox = detection.location_data.relative_bounding_box
                h, w, c = image.shape
                boundBox = int(bBox.xmin * w), int(bBox.ymin * h), int(bBox.width * w), int(bBox.height * h)
                cv2.putText(image, f'{int(detection.score[0]*100)}%', (boundBox[0], boundBox[1] - 20), cv2.FONT_HERSHEY_SIMPLEX, 1, (0,255,0), 2)
        end = time.time()
        totalTime = end - start
        if totalTime:
            fps = 1 / totalTime
            cv2.putText(image, f'FPS: {int(fps)}', (20,70), cv2.FONT_HERSHEY_SIMPLEX, 1, (0,255,0), 2)
        end = start
        # image = resize(image, 0.5)
        cv2.imshow('Face Detection', image)
        if cv2.waitKey(5) & 0xFF == 27:
            break
cap.release()
cv2.destroyAllWindows()