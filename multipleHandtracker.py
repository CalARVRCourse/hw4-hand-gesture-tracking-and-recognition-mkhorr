import cv2
import numpy as np
import pyautogui
import time

cam = cv2.VideoCapture(0)
windowWidth = 640
windowHeight = 480
cam.set(3, windowWidth)
cam.set(4, windowHeight)
cv2.namedWindow("Display")
cv2.namedWindow("Trackbars")

#tuning parameters
p = {
    "low_H": [0, 255],
    "high_H": [255, 255], #255, default: 25
    "low_S": [52, 255], #60, default: 40
    "low_V": [0, 255],
    "low_Y": [0, 255],
    "low_Cr": [135, 255], #day: 135, default: 138
    "low_Cb": [143, 255], #day: 80, night: 143, default: 67
    "kernel_param": [10, 20, lambda x: 1 if x == 0 else x], #10, default: 7
    "blur_param": [3, 20, lambda x: x + 1 if x % 2 == 0 else x],
    "min_binary_value": [0, 255],
    "max_binary_value": [255, 255],
    "threshold": [0, 100],
    "angle_threshold":[16, 140, lambda x: x/4]
}

def set_param(param):
    def setParam(x):
        global p
        if len(p[param]) == 3:
            p[param][0] = p[param][2](x)
        else:
            p[param][0] = x
    return setParam

for param in p.keys():
    cv2.createTrackbar(param, "Trackbars" , p[param][0], p[param][1], set_param(param))


def isActive(ls):
    count = 0
    for v in ls:
        if v == None:
            count += 1
    return count < len(ls) / 2

def avgTuples(ls):
    avg = None
    count = 0
    for tup in ls:
        if tup == None:
            continue
        count += 1
        if avg == None:
            avg = list(tup)
            continue
        for i in range(len(tup)):
            avg[i] += tup[i]
    if avg == None:
        return None
    for i in range(len(avg)):
        avg[i] = avg[i] / count
    return tuple(avg)

def isOnBorder(point):
    hPad = 20
    vPad = 30
    top = point[1] >= windowHeight - hPad
    bot = point[1] <= hPad
    left = point[0] <= vPad
    right = point[0] >= windowWidth - vPad
    return top or bot or left or right

def angleDifference(a1, a2):
    a1 = a1 - a2
    if a1 < 0:
        a1 += 2*np.pi
    a2 = a2 - a2
    diff = a1 - a2
    if diff > np.pi:
        return -(2*np.pi - diff)
    else:
        return diff

def resetGesture(ls):
    for i in range(len(ls)):
        ls[i] = None



lak,rak,uak,dak = (False,False,False,False)

def pressArrowKeys(v):
    global lak,rak,uak,dak
    l,r,u,d = (False,False,False,False)
    angle = np.arctan2(-v[1], -v[0])
    #print(angle*180/np.pi)
    if angle >= -3*np.pi/8 and angle <= 3*np.pi/8:
        r = True
    if angle >= np.pi/8 and angle <= 7*np.pi/8:
        u = True
    if angle >= 5*np.pi/8 or angle <= -5*np.pi/8:
        l = True
    if angle >= -7*np.pi/8 and angle <= -np.pi/8:
        d = True
    
    if l != lak:
        lak = l
        pyautogui.keyDown('left') if l else pyautogui.keyUp('left')
    if r != rak:
        rak = r
        pyautogui.keyDown('right') if r else pyautogui.keyUp('right')
    if u != uak:
        uak = u
        pyautogui.keyDown('up') if u else pyautogui.keyUp('up')
    if d != dak:
        dak = d
        pyautogui.keyDown('down') if d else pyautogui.keyUp('down')

def releaseArrowKeys():
    global lak,rak,uak,dak
    if lak:
        lak = False
        pyautogui.keyUp('left')
    if rak:
        rak = False
        pyautogui.keyUp('right')
    if uak:
        uak = False
        pyautogui.keyUp('up')
    if dak:
        dak = False
        pyautogui.keyUp('down')


#part 1
def apply_skin_mask(frame):
    lower_HSV = np.array([p["low_H"][0], p["low_S"][0], p["low_V"][0]], dtype = "uint8")
    upper_HSV = np.array([p["high_H"][0], 255, 255], dtype = "uint8")  
  
    convertedHSV = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)  
    skinMaskHSV = cv2.inRange(convertedHSV, lower_HSV, upper_HSV)
  
    lower_YCrCb = np.array((p["low_Y"][0], p["low_Cr"][0], p["low_Cb"][0]), dtype = "uint8")  
    upper_YCrCb = np.array((255, 173, 133), dtype = "uint8")  
      
    convertedYCrCb = cv2.cvtColor(frame, cv2.COLOR_BGR2YCrCb)  
    skinMaskYCrCb = cv2.inRange(convertedYCrCb, lower_YCrCb, upper_YCrCb)  
  
    skinMask = cv2.add(skinMaskHSV,skinMaskYCrCb)


    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE,
        (p["kernel_param"][0], p["kernel_param"][0]))  
    skinMask = cv2.erode(skinMask, kernel, iterations = 2)  
    skinMask = cv2.dilate(skinMask, kernel, iterations = 2)  
  
    # blur the mask to help remove noise, then apply the  
    # mask to the frame  
    skinMask = cv2.GaussianBlur(skinMask, (p["blur_param"][0], p["blur_param"][0]), 0) 
    skin = cv2.bitwise_and(frame, frame, mask = skinMask)
    return skin


#part 2
def binarize_img(img):
    gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
    return cv2.threshold(gray, p["min_binary_value"][0], p["max_binary_value"][0], cv2.THRESH_BINARY_INV )

def findEllipse(thresh):
    ret, markers, stats, centroids = cv2.connectedComponentsWithStats(thresh,ltype=cv2.CV_16U)  
    markers = np.array(markers, dtype=np.uint8)
    label_hue = np.uint8(179*markers/np.max(markers))
    blank_ch = 255*np.ones_like(label_hue)
    labeled_img = cv2.merge([label_hue, blank_ch, blank_ch])
    labeled_img = cv2.cvtColor(labeled_img,cv2.COLOR_HSV2BGR)
    labeled_img[label_hue==0] = 0

    statsSortedByArea = stats[np.argsort(stats[:, 4])]
    if (ret>2):
        try:
            roi = statsSortedByArea[-3][0:4]  
            x, y, w, h = roi  
            subImg = labeled_img[y:y+h, x:x+w]  
            subImg = cv2.cvtColor(subImg, cv2.COLOR_BGR2GRAY);  
            _, contours, _ = cv2.findContours(subImg, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)  
            maxCntLength = 0  
            for i in range(0,len(contours)):
                cntLength = len(contours[i])
                if(cntLength>maxCntLength):
                    cnt = contours[i]
                    maxCntLength = cntLength  
            if(maxCntLength>=5):  
                ellipseParam = cv2.fitEllipse(cnt)  
                subImg = cv2.cvtColor(subImg, cv2.COLOR_GRAY2RGB)
                subImg = cv2.ellipse(subImg,ellipseParam,(0,255,0),2)
              
            subImg = cv2.resize(subImg, (0,0), fx=3, fy=3)

            ellipse = cv2.fitEllipse(cnt)
            (x,y),(MA,ma),angle = ellipse
            if (MA < 20 or ma < 20 or MA/ma > 4 or ma/MA > 4):
                return labeled_img, None, None, None
            print(f"({x:8.4f}, {y:8.4f}), ({MA:8.4f}, {ma:8.4f}), {angle:8.4f}")
            return labeled_img, subImg, ellipse, roi

        except:
            print("No hand found")
            return labeled_img, None, None, None
    return labeled_img, None, None, None



#Part 4

#sliding window for when the gestures is detected
window_index = 0
#handIndex
point_window = [[None]*10, [None]*10]
point_start = [None, None]

shaka_window = [[None]*10, [None]*10]
shaka_start = [None, None]

finger3_window = [[None]*10, [None]*10]
finger3_start = [None, None]

circle_window = [[None]*10, [None]*10]
circle_start = [None, None]
circle_prev = [None, None]

area_window = [[None]*10, [None]*10]
area_press = [False, False]

zoom_window = [[None]*10, [None]*10]
previous_angle = [None, None]

countdown = [False, False]
countdown_start_frame = [0, 0]
last_finger = [-1 ,-1]

fps = 30
prevTime = int(time.time()) / 1000

while True:
    ret, frame = cam.read()
    if not ret:
        break

    currentTime = int(time.time()) / 1000
    timeDelta = currentTime - prevTime
    prevTime = currentTime
    if timeDelta < 1 / fps:
        time.sleep((1 / fps) - timeDelta)

    skin = apply_skin_mask(frame)
    gray = cv2.cvtColor(skin,cv2.COLOR_BGR2GRAY)
    ret, thresh = cv2.threshold(gray, p["min_binary_value"][0], p["max_binary_value"][0], cv2.THRESH_BINARY_INV )
    ret, thresh = binarize_img(skin)
    labeled_img, subImg, ellipse, roi = findEllipse(thresh)

    thresh = cv2.bitwise_not(thresh)
    thresholdedHandImage = cv2.cvtColor(thresh,cv2.COLOR_GRAY2RGB)
    _, contours, _ = cv2.findContours(thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    contours = sorted(contours,key=cv2.contourArea,reverse=True)
    
    if len(contours)>0:
        fingerCount = 0
        fingertipPositions = []
        shaka = False
        finger_angle = 0
        largestContour = contours[0]
        hull = cv2.convexHull(largestContour, returnPoints = False)
        contours1 = contours[:min(2,len(contours))]
        if len(contours) > 1:
            secondLargestContour = contours[1]
            hull1 = cv2.convexHull(secondLargestContour, returnPoints = False)

            M1 = cv2.moments(largestContour)
            x1 = int(M1["m10"] / M1["m00"])
            y1 = int(M1["m01"] / M1["m00"])
            try:
                M2 = cv2.moments(secondLargestContour)
                x2 = int(M2["m10"] / M2["m00"])
                y2 = int(M2["m01"] / M2["m00"])
            except:
                contours1 = contours[:1]

            #always process left hand first
            if (x1 > x2):
                temp = largestContour
                largestContour = secondLargestContour
                secondLargestContour = temp
                temp = hull
                hull = hull1
                hull1 = temp
                contours1.reverse()
        
        for handIndex, cnt in enumerate(contours1):
            if handIndex == 1:
                hull = hull1
                largestContour = secondLargestContour
            defects = cv2.convexityDefects(cnt,hull)
            if(not isinstance(defects,type(None))):
                for i in range(defects.shape[0]):
                    s,e,f,d = defects[i,0]
                    start = tuple(cnt[s][0])
                    end = tuple(cnt[e][0])
                    far = tuple(cnt[f][0])

                    c_squared = (end[0] - start[0]) ** 2 + (end[1] - start[1]) ** 2
                    a_squared = (far[0] - start[0]) ** 2 + (far[1] - start[1]) ** 2
                    b_squared = (end[0] - far[0]) ** 2 + (end[1] - far[1]) ** 2
                    angle = np.arccos((a_squared + b_squared  - c_squared ) / (2 * np.sqrt(a_squared * b_squared )))    
                      
                    if angle <= np.pi * (90 / 180):
                        fingerCount += 1
                        finger_angle = angle 
                        cv2.circle(thresholdedHandImage, far, 4, [0, 0, 255], -1)

                    cv2.line(thresholdedHandImage,start,end,[0,255,0],2)
                
                # addressing the case where there is only one finger
                if fingerCount == 0:
                    for i in range(defects.shape[0] * 2):
                        s,e,f,d = defects[i % defects.shape[0],0]
                        start = tuple(cnt[s][0])
                        end = tuple(cnt[e][0])
                        if i == 0:
                            prev_start = start
                            continue

                        c_squared = (end[0] - prev_start[0]) ** 2 + (end[1] - prev_start[1]) ** 2
                        a_squared = (start[0] - prev_start[0]) ** 2 + (start[1] - prev_start[1]) ** 2
                        b_squared = (end[0] - start[0]) ** 2 + (end[1] - start[1]) ** 2
                        angle = np.arccos((a_squared + b_squared  - c_squared ) / (2 * np.sqrt(a_squared * b_squared )))
                        if b_squared < 500:
                            continue

                        if angle <= np.pi * (90 / 180) and not isOnBorder(start):
                            fingerCount += 1
                            fingertipPositions.append(start)
                        prev_start = start
                    if fingerCount > 2:
                        fingerCount = 2
                        shaka = True
                    elif fingerCount != 0:
                        fingerCount = 1
                else:
                    fingerCount += 1
    

            print(fingerCount)

            # gestures

            #punch to click
            if fingerCount == 0 and ellipse is None and largestContour is not None and not countdown[handIndex]:
                area = cv2.contourArea(largestContour)
                if area > 60000:
                    print("AREA: ", area)
                    area_window[handIndex][window_index % len(area_window[handIndex])] = True
                else:
                    area_window[handIndex][window_index % len(area_window[handIndex])] = None
            else:
                area_window[handIndex][window_index % len(area_window[handIndex])] = None
            if isActive(area_window[handIndex]):
                if area_press[handIndex] == False:
                    print("CLICK")
                    pyautogui.click()
                    area_press[handIndex] = True
            else:
                area_press[handIndex] = False


            # point to pan map
            if fingerCount == 1 and ellipse is None and not countdown[handIndex]:
                M = cv2.moments(largestContour)
                x = int(M["m10"] / M["m00"])
                y = int(M["m01"] / M["m00"])
                px, py = fingertipPositions[0]
                magnitude = ((px - x)**2 + (py - y)**2)**.5
                pointerDirection = ((px - x) / magnitude, (py - y) / magnitude)
                point_window[handIndex][window_index % len(point_window[handIndex])] = pointerDirection
            else:
                point_window[handIndex][window_index % len(point_window[handIndex])] = None
            if isActive(point_window[handIndex]):
                #currentPosition = pyautogui.position()
                pointerDirection = avgTuples(point_window[handIndex])
                #scale = 30.0
                #cx, cy = (currentPosition[0] - scale * pointerDirection[0],
                #          currentPosition[1] + scale * pointerDirection[1])
                #pyautogui.moveTo(cx, cy)
                pressArrowKeys(pointerDirection)
            else:
                releaseArrowKeys()
    
            # shaka to rotate
            if shaka and ellipse is None and not countdown[handIndex]:
                leftX, leftY = fingertipPositions[0]
                rightX, rightY = fingertipPositions[1]
                shaka_window[handIndex][window_index % len(shaka_window[handIndex])] = (leftX, leftY, rightX, rightY)
            else:
                shaka_window[handIndex][window_index % len(shaka_window[handIndex])] = None
            if isActive(shaka_window[handIndex]):
                avg = avgTuples(shaka_window[handIndex])
                if shaka_start[handIndex] == None:
                    shaka_start[handIndex] = (avg, pyautogui.position())
                    pyautogui.keyDown('shift')
                    pyautogui.mouseDown()
                else:
                    lX, lY, rX, rY = avg
                    (slX, slY, srX, srY), (sx, sy) = shaka_start[handIndex]

                    midX = (lX + rX) / 2
                    midY = (lY + rY) / 2
                    sMidX = (slX + srX) / 2
                    sMidY = (slY + srY) / 2

                    angle = np.arctan2(rY - midY, rX - midX)
                    sAngle = np.arctan2(srY - sMidY, srX - sMidX)

                    angleDiff = angleDifference(angle, sAngle)
                    factor = 500.0
                    pyautogui.moveTo(sx + factor * angleDiff, sy)
            else:
                if shaka_start[handIndex] is not None:
                    shaka_start[handIndex] = None
                    pyautogui.keyUp('shift')
                    pyautogui.mouseUp()
    
            # draw circle gesture
            if ellipse is not None and not countdown[handIndex]:
                print("ELLIPSE")
                (xc,yc),(MA,ma),angle = ellipse
                x, y, w, h = roi
                circle_window[handIndex][window_index % len(circle_window[handIndex])] = (x+xc,y+yc)
                #cv2.circle(thresholdedHandImage, (int(x+xc),int(y+yc)), 4, [0, 0, 255], -1)
            else:
                circle_window[handIndex][window_index % len(circle_window[handIndex])] = None
            if isActive(circle_window[handIndex]):
                avg = avgTuples(circle_window[handIndex])
                if circle_start[handIndex] is None:
                    circle_start[handIndex] = avg
                    circle_prev[handIndex] = avg
                else:
                    scX = windowWidth / 2
                    scY = windowHeight / 2
                    #cv2.circle(thresholdedHandImage, (int(scX),int(scY)), 4, [0, 255, 0], -1)
                    start_angle = np.arctan2(circle_start[handIndex][1] - scY, circle_start[handIndex][0] - scX)
                    prev_angle = np.arctan2(circle_prev[handIndex][1] - scY, circle_prev[handIndex][0] - scX)
                    angle = np.arctan2(avg[1] - scY, avg[0] - scX)
                    print(angle*180/np.pi)
                    if angleDifference(angle, prev_angle) < 0:
                        circle_window[handIndex][window_index % len(circle_window[handIndex])] = None
                    elif angleDifference(prev_angle, start_angle) < 0 and angleDifference(angle, start_angle) > 0:
                        resetGesture(circle_window[handIndex])
                        pyautogui.press('r')
                    else:
                        circle_prev[handIndex] = avg
            else:
                if circle_start[handIndex] is not None:
                    circle_start[handIndex] = None
                    circle_prev[handIndex] = None
    
            #countdown[handIndex]
            if countdown[handIndex] or fingerCount == 5:
                if fingerCount == 0:
                    countdown[handIndex] = False
                    pyautogui.keyDown('command')
                    pyautogui.press('w')
                    pyautogui.keyUp('command')
                elif fingerCount == 5:
                    countdown[handIndex] = True
                    last_finger[handIndex] = 5
                    countdown_start_frame[handIndex] = window_index
                elif fingerCount == last_finger[handIndex] or fingerCount == last_finger[handIndex] - 1:
                    last_finger[handIndex] = fingerCount
                    if window_index - countdown_start_frame[handIndex] > fps:
                        countdown[handIndex] = False
                else:
                    countdown[handIndex] = False

            #3-finger cursor
            if fingerCount == 3 and ellipse is None and not countdown[handIndex]:
                M = cv2.moments(largestContour)
                x = int(M["m10"] / M["m00"])
                y = int(M["m01"] / M["m00"])
                finger3_window[handIndex][window_index % len(finger3_window[handIndex])] = (x, y)
            else:
                finger3_window[handIndex][window_index % len(finger3_window[handIndex])] = None
            if isActive(finger3_window[handIndex]):
                avg = avgTuples(finger3_window[handIndex])
                if finger3_start[handIndex] == None:
                    finger3_start[handIndex] = (avg, pyautogui.position())
                else:
                    (sX, sY), (cX, cY) = finger3_start[handIndex]
                    x, y = avg
                    scale = 3.0
                    pyautogui.moveTo(cX - scale * (x - sX), cY + scale * (y - sY))
            else:
                if finger3_start[handIndex] is not None:
                    finger3_start[handIndex] = None

            #2-finger zoom
            if fingerCount == 2 and not shaka and ellipse is None and not countdown[handIndex]:
                zoom_window[window_index % len(zoom_window)] = (finger_angle,)
            else:
                zoom_window[window_index % len(zoom_window)] = None
            if isActive(zoom_window):
                #avg, = avgTuples(zoom_window)
                avg = finger_angle
                if previous_angle[handIndex] is None:
                    previous_angle[handIndex] = avg
                else:
                    angleDiff = angleDifference(avg, previous_angle[handIndex])
                    threshold = np.pi * (p["angle_threshold"][0]/180)
                    clicks = 20
                    #print(angleDiff*180/np.pi)

                    if angleDiff > threshold:
                        pyautogui.scroll(clicks)
                    elif angleDiff < -threshold:
                        pyautogui.scroll(-clicks)
                    previous_angle[handIndex] = avg
            else:
                previous_angle[handIndex] = None







    window_index += 1

    # display the current image
    cv2.imshow("Display", cv2.flip(thresholdedHandImage, 1))
    # wait for 1ms or key press
    k = cv2.waitKey(1) #k is the key pressed
    if k == 27 or k==113:  #27, 113 are ascii for escape and q respectively
        #exit
        break









cam.release()