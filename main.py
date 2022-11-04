import cv2
import mediapipe as mp
import time
import math
import numpy as np

# access camera
cap = cv2.VideoCapture(0)
# hand detection model by mediapipe
mpHands = mp.solutions.hands
# method details: ref docs of mediapipe
hands = mpHands.Hands(max_num_hands=1)

# draw hand landmarks, and its style settings
mpDraw = mp.solutions.drawing_utils
handLmsStyle = mpDraw.DrawingSpec(color=(255,255,255), thickness=5)
handConStyle = mpDraw.DrawingSpec(color=(255,0,0), thickness=3)

# FPS
# previous-time
pTime = 0
# current-time
cTime = 0



while True:
    ret, frame = cap.read()

    if ret:
        # all frames needs to convert to RGB (opencv(BGR)->mediapipe(RGB))
        imgRGB = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        result = hands.process(imgRGB)

        # get the height and width size of every frame
        imgHeight = frame.shape[0]
        imgWidth = frame.shape[1]

       # if landmarks coordinate detected
        if result.multi_hand_landmarks:
            
            # draw landmark
            for handLms in result.multi_hand_landmarks:
                mpDraw.draw_landmarks(frame, handLms, mpHands.HAND_CONNECTIONS, handLmsStyle, handConStyle)
                location = []

                # draw landmark id
                for id, lm in enumerate(handLms.landmark):
                    # landmark position coordinate setting
                    xPosition = np.int(lm.x * imgWidth)
                    yPosition = np.int(lm.y * imgHeight)

                    # put id string next to the landmark
                    cv2.putText(frame, str(id), (xPosition-30,yPosition+5), cv2.FONT_ITALIC, 0.3, (255,255,255), 1)

                    # coordinate of each id in a list
                    coor = []
                    coor.append(xPosition)
                    coor.append(yPosition)
                    location.append(coor)
                #print(location)

            finger4 = location[4]
            finger8 = location[8]
            fingerDistance = math.sqrt(((finger4[0]-finger8[0])**2)+((finger4[1]-finger8[1])**2))
            #print(int(fingerDistance))

            # if distance <70, show h/l landmark circle and yellow circle
            if fingerDistance < 70:
                thumb = cv2.circle(frame, (finger4[0], finger4[1]), 8, (0, 255, 0), cv2.FILLED)
                forefinger = cv2.circle(frame, (finger8[0], finger8[1]), 8, (0, 255, 0), cv2.FILLED)
                volumeLine = cv2.line(frame, (finger4[0],finger4[1]), (finger8[0],finger8[1]), (255,255,255), 1)
                centerCir = cv2.circle(frame, (int((finger4[0]+finger8[0])/2),int((finger4[1]+finger8[1])/2)), 8, (0, 255, 255), cv2.FILLED)
                
            
        # FPS
        cTime = time.time()
        fps = 1 / (cTime - pTime)
        pTime = cTime
        cv2.putText(frame, f'FPS: {int(fps)}', (30,50), cv2.FONT_ITALIC, 0.5, (128,128,128), 1)



        cv2.imshow('Capture', frame)

    if cv2.waitKey(20) == ord('q'):
        break