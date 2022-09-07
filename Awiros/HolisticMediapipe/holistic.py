import cv2
import mediapipe as mp
import time

mp_drawing = mp.solutions.drawing_utils
mp_holistic = mp.solutions.holistic

cap = cv2.VideoCapture(0)
with mp_holistic.Holistic(min_detection_confidence=0.5, min_tracking_confidence=0.5) as holistic:
    s = time.time()
    while cap.isOpened():
        ret, frame = cap.read()
        
        # BGR 2 RGB
        image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        
        # Flip on Horizontal Axis
        image = cv2.flip(image, 1)
        
        # Process
        results = holistic.process(image)
        
        # RGB 2 BGR
        image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
        
        # Draw face landmarks
        # mp_drawing.draw_landmarks(image, results.face_landmarks, mp_holistic.FACEMESH_CONTOURS,
        #                          mp_drawing.DrawingSpec(color = (25, 0, 255), thickness = 1, circle_radius = 1), 
        #                           mp_drawing.DrawingSpec(color = (240, 0, 0), thickness = 1, circle_radius = 1)
        #                          ) 
        
        #Right Hand
        mp_drawing.draw_landmarks(image, results.right_hand_landmarks, mp_holistic.HAND_CONNECTIONS, 
                                  mp_drawing.DrawingSpec(color = (255, 0, 255), thickness = 2, circle_radius = 4), 
                                  mp_drawing.DrawingSpec(color = (240, 0, 0), thickness = 2, circle_radius = 2)
                                 ) 
        
        # Left Hand
        mp_drawing.draw_landmarks(image, results.left_hand_landmarks, mp_holistic.HAND_CONNECTIONS,
                                 mp_drawing.DrawingSpec(color = (255, 0, 255), thickness = 2, circle_radius = 4), 
                                  mp_drawing.DrawingSpec(color = (240, 0, 0), thickness = 2, circle_radius = 2)
                                 ) 
        
        #Pose Detections
        mp_drawing.draw_landmarks(image, results.pose_landmarks, mp_holistic.POSE_CONNECTIONS,
                                 mp_drawing.DrawingSpec(color = (255, 0, 255), thickness = 2, circle_radius = 4), 
                                  mp_drawing.DrawingSpec(color = (240, 0, 0), thickness = 2, circle_radius = 2)
                                 ) 
        e = time.time()
        fps=1/(e-s)
        cv2.putText(image, f'FPS : {int(fps)}', (20, 100), cv2.FONT_HERSHEY_DUPLEX, 1, (255, 0, 0), 2)
        s = e
        cv2.imshow("Hoslistic Model Detection", image)
        
        k = cv2.waitKey(10)
        if k == ord('q'):
            break
cap.release()
cv2.destroyAllWindows()