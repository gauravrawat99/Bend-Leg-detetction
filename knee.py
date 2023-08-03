import cv2
import time
import mediapipe as mp
import numpy as np
mp_drawing = mp.solutions.drawing_utils
mp_pose = mp.solutions.pose


def calculate_angle(a,b,c):
    a = np.array(a) # First
    b = np.array(b) # Mid
    c = np.array(c) # End
    
    radians = np.arctan2(c[1]-b[1], c[0]-b[0]) - np.arctan2(a[1]-b[1], a[0]-b[0])
    angle = np.abs(radians*180.0/np.pi)
    
    if angle >180.0:
        angle = 360-angle
        
    return angle 


t1 = 0
t2 = 0
flag=-1
i=30
j=30
count =0
timer = 0
k=192


# angle = None 'C:/Users/rawat/Desktop/opencv/KneeBend.mp4'
font = cv2.FONT_HERSHEY_SIMPLEX
# tip = str(Keep)
cap = cv2.VideoCapture('C:/Users/rawat/Desktop/opencv/KneeBend.mp4')
fourcc = cv2.VideoWriter_fourcc(*'XVID')
out = cv2.VideoWriter('output.avi', fourcc, 20.0, (640,480))
with mp_pose.Pose(min_detection_confidence=0.9, min_tracking_confidence=0.9) as pose:
    while cap.isOpened():
        ret, frame = cap.read()
        # Recolor image to RGB
        image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        image.flags.writeable = False
      
        # Make detection
        results = pose.process(image)
    
        # Recolor back to BGR
        image.flags.writeable = True
        image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
        
        # Extract landmarks
        try:
            landmarks = results.pose_landmarks.landmark
            
            # Get coordinates
            hip = [landmarks[mp_pose.PoseLandmark.LEFT_HIP.value].x,landmarks[mp_pose.PoseLandmark.LEFT_HIP.value].y]
            knee = [landmarks[mp_pose.PoseLandmark.LEFT_KNEE.value].x,landmarks[mp_pose.PoseLandmark.LEFT_KNEE.value].y]
            ankle = [landmarks[mp_pose.PoseLandmark.LEFT_ANKLE.value].x,landmarks[mp_pose.PoseLandmark.LEFT_ANKLE.value].y]
            
            # Calculate angle
            angle = calculate_angle(hip, knee, ankle)
            
#             if angle <150:
#                 t = 8
#                 while t!=0:
#                     cv2.putText(image, str(t), (20, 35), font, 0.65, (0, 0, 255), 2, cv2.LINE_AA)
#                     t=t-1
            
            if angle <140 and t1==0:
                t1=time.time()
                
                
            if t1>0 and angle >=140 and t2==0:
                t2=time.time()
                i=30
                j=30
                timer=0
                k=192
                 
            if t1>0 and t2>0:
                if t2-t1<8 and angle >140:
                        flag=1
                        t1=0
                        t2=0
                       
                elif t2-t1>=8:
                        flag=0
                        count = count + 1
                        t1=0
                        t2=0
                
            if flag==1 and i>0:
                cv2.putText(image, "Keep your knee bent", (20, 35), font, 1, (0, 0, 255), 2, cv2.LINE_AA)                   
                i=i-1
            if flag == 0 and j>0:
                cv2.putText(image, "good job", (20, 35), font, 1, (0, 0, 255), 2, cv2.LINE_AA)
                j=j-1
                 
                
                        
                        
        except:
            pass
        #to put text to check angle value on the screen
#         cv2.putText(image, str(angle), 
#                            tuple(np.multiply(elbow, [640, 480]).astype(int)), 
#                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2, cv2.LINE_AA
#                                 )
        if k>0 and angle <140:
            if k%24==0:
                timer = timer + 1
            cv2.putText(image, str(timer), (135, 95), font, 0.65, (0, 0, 255), 2, cv2.LINE_AA)
            k=k-1
        cv2.putText(image, str(count), (135, 65), font, 0.65, (0, 0, 255), 2, cv2.LINE_AA)
        cv2.putText(image, "counter = ", (15, 65), font, 0.65, (0, 0, 255), 2, cv2.LINE_AA)
#         cv2.putText(image, str(timer), (135, 95), font, 0.65, (0, 0, 255), 2, cv2.LINE_AA)
        cv2.putText(image, "timer = ", (15, 95), font, 0.65, (0, 0, 255), 2, cv2.LINE_AA)
       
        mp_drawing.draw_landmarks(image, results.pose_landmarks, mp_pose.POSE_CONNECTIONS,
                                mp_drawing.DrawingSpec(color=(245,117,66), thickness=2, circle_radius=2), 
                                mp_drawing.DrawingSpec(color=(245,66,230), thickness=2, circle_radius=2) 
                                 )               
        
#         cv2.imshow('Mediapipe Feed', image)
        if ret==True:


            out.write(image)
  

        if cv2.waitKey(10) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()
out.release()
result.release()