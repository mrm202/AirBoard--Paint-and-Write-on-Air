import cv2
import numpy as np
from collections import deque

def empty(x):
   pass


# Creating the trackbars needed to track the object through which we want to write
cv2.namedWindow("TrackBars")
cv2.resizeWindow("TrackBars",640,240)
cv2.createTrackbar("Hue Min","TrackBars",64,180,empty)
cv2.createTrackbar("Hue Max","TrackBars",153,180,empty)
cv2.createTrackbar("Sat Min","TrackBars",72,255,empty)
cv2.createTrackbar("Sat Max","TrackBars",255,255,empty)
cv2.createTrackbar("Val Min","TrackBars",49,255,empty)
cv2.createTrackbar("Val Max","TrackBars",255,255,empty)


# Arrays to handle colour points of different colours
bluepoints = [deque(maxlen=1024)]
greenpoints = [deque(maxlen=1024)]
redpoints = [deque(maxlen=1024)]
yellowpoints = [deque(maxlen=1024)]

#Assign index values for colors
blue_index = 0
green_index = 0
red_index = 0
yellow_index = 0

kernel = np.ones((5,5),np.uint8)

colors = [(255, 0, 0), (0, 255, 0), (0, 0, 255), (0, 255, 255)]
colorIndex = 0

#AirBoard Window
airBoard = np.zeros((471,636,3)) + 255
airBoard = cv2.rectangle(airBoard, (50,1), (150,70), (0,0,0), -1)
airBoard = cv2.rectangle(airBoard, (160,1), (260,70), colors[0], -1)
airBoard = cv2.rectangle(airBoard, (270,1), (370,70), colors[1], -1)
airBoard = cv2.rectangle(airBoard, (380,1), (480,70), colors[2], -1)
airBoard = cv2.rectangle(airBoard, (490,1), (590,70), colors[3], -1)

cv2.putText(airBoard, "CLEAR", (60, 35), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2, cv2.LINE_AA)
cv2.putText(airBoard, "BLUE", (170, 35), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2, cv2.LINE_AA)
cv2.putText(airBoard, "GREEN", (280, 35), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2, cv2.LINE_AA)
cv2.putText(airBoard, "RED", (390, 35), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2, cv2.LINE_AA)
cv2.putText(airBoard, "YELLOW", (500, 35), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255,255,0), 2, cv2.LINE_AA)

cap = cv2.VideoCapture(0)
while True:
    success, img = cap.read()
    #Flipping the image for convenience
    img = cv2.flip(img, 1)
    imgHSV = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)


    hue_min = cv2.getTrackbarPos("Hue Min", "TrackBars")
    hue_max = cv2.getTrackbarPos("Hue Max", "TrackBars")
    sat_min = cv2.getTrackbarPos("Sat Min", "TrackBars")
    sat_max = cv2.getTrackbarPos("Sat Max", "TrackBars")
    val_min = cv2.getTrackbarPos("Val Min", "TrackBars")
    val_max = cv2.getTrackbarPos("Val Max", "TrackBars")
    lower_hsv = np.array([hue_min,sat_min,val_min])
    upper_hsv = np.array([hue_max,sat_max,val_max])

    img = cv2.rectangle(img, (50,1), (150,70), (0,0,0), -1)
    img = cv2.rectangle(img, (160,1), (260,70), colors[0], -1)
    img = cv2.rectangle(img, (270,1), (370,70), colors[1], -1)
    img = cv2.rectangle(img, (380,1), (480,70), colors[2], -1)
    img = cv2.rectangle(img, (490,1), (590,70), colors[3], -1)
    cv2.putText(img, "CLEAR", (60, 35), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2, cv2.LINE_AA)
    cv2.putText(img, "BLUE", (170, 35), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2, cv2.LINE_AA)
    cv2.putText(img, "GREEN", (280, 35), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2, cv2.LINE_AA)
    cv2.putText(img, "RED", (390, 35), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2, cv2.LINE_AA)
    cv2.putText(img, "YELLOW", (500, 35), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255,255,0), 2, cv2.LINE_AA)

    Mask = cv2.inRange(imgHSV, lower_hsv, upper_hsv)
    Mask = cv2.erode(Mask, kernel, iterations=1)
    Mask = cv2.morphologyEx(Mask, cv2.MORPH_OPEN, kernel)
    Mask = cv2.dilate(Mask, kernel, iterations=1)

    cnts, hierarchy = cv2.findContours(Mask.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    center = None

    # Checking whether contours have been formed
    if len(cnts) > 0:
    # sorting the contours to find the biggest contour
        cnt = sorted(cnts, key = cv2.contourArea, reverse = True)[0]
        # Get the radius of the circle around the biggest contour
        ((x, y), radius) = cv2.minEnclosingCircle(cnt)
        # Draw the circle around the contour
        cv2.circle(img, (int(x), int(y)), int(radius), (0, 255, 255), 2)
        # Calculating the center of the detected contour
        M = cv2.moments(cnt)
        center = (int(M['m10'] / M['m00']), int(M['m01'] / M['m00']))

        #Check if any button above the screen is clicked/object goes over
        if center[1] <= 70:
            if 50 <= center[0] <= 150:
                bluepoints = [deque(maxlen=512)]
                greenpoints = [deque(maxlen=512)]
                redpoints = [deque(maxlen=512)]
                yellowpoints = [deque(maxlen=512)] # This will clear screen

                blue_index = 0
                green_index = 0
                red_index = 0
                yellow_index = 0

                airBoard[75:,:,:] = 255
            elif 160 <= center[0] <= 260:
                    colorIndex = 0 # This will start writing in blue
            elif 270 <= center[0] <= 370:
                    colorIndex = 1 # This will start writing in green
            elif 380 <= center[0] <= 480:
                    colorIndex = 2 # This will start writing in red
            elif 490 <= center[0] <= 590:
                    colorIndex = 3 # This will start writing in yellow
        else :
            if colorIndex == 0:
                bluepoints[blue_index].appendleft(center)
            elif colorIndex == 1:
                greenpoints[green_index].appendleft(center)
            elif colorIndex == 2:
                redpoints[red_index].appendleft(center)
            elif colorIndex == 3:
                yellowpoints[yellow_index].appendleft(center)
    else:
        bluepoints.append(deque(maxlen=512))
        blue_index += 1
        greenpoints.append(deque(maxlen=512))
        green_index += 1
        redpoints.append(deque(maxlen=512))
        red_index += 1
        yellowpoints.append(deque(maxlen=512))
        yellow_index += 1

    points = [bluepoints, greenpoints, redpoints, yellowpoints]
    for i in range(len(points)):
        for j in range(len(points[i])):
            for k in range(1, len(points[i][j])):
                if points[i][j][k - 1] is None or points[i][j][k] is None:
                    continue
                cv2.line(img, points[i][j][k - 1], points[i][j][k], colors[i], 2)
                cv2.line(airBoard, points[i][j][k - 1], points[i][j][k], colors[i], 2)


    cv2.imshow("Tracking", img)
    cv2.imshow("Mask",Mask)
    cv2.imshow("AirBoard", airBoard)
    


    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

cap.release()
cv2.destroyAllWindows()
