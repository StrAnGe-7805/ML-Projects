import cv2
import numpy as np
import matplotlib.pyplot as plt

def canny(lane_image):   #function to convert an image into edge detected image(image with edges)
    gray = cv2.cvtColor(lane_image,cv2.COLOR_RGB2GRAY)   #Converts normal image to grey image(almost black and white)
    blur = cv2.GaussianBlur(gray,(5,5),0)    # Makes the image smooth 
    cany = cv2.Canny(blur,50,150)  # Detects the edged and converts edges to white and the whole other image to black
    return cany

def region_of_intrests(image):      # returns an image with a shape of desired polygon
    height = image.shape[0]
    polygon = np.array([[(200,height),(1100,height),(550,250)]])  # shape of polygon
    mask = np.zeros_like(image)     # Creates a black image with same dimensions as desired image 
    cv2.fillPoly(mask,polygon,255)   # Creates a white region in the mask with the shape of polygon
    masked_image = cv2.bitwise_and(image,mask)  # Copys the polygon shape of image into mask
    return masked_image

def make_coordinates(image , line_parameters):
    slope , intercept = line_parameters
    y1 = image.shape[0]
    y2 = int(y1*(3/5))
    x1 = int((y1 - intercept)/slope)
    x2 = int((y2 - intercept)/slope)
    return np.array([x1 , y1 , x2 , y2])

def average_slpoe_intercept(image , lines):
    left_fit = []
    right_fit = []
    for line in lines:
        x1 , y1 , x2 , y2  = line.reshape(4)
        parameters = np.polyfit((x1,x2),(y1,y2),1)
        slope = parameters[0]
        intercept = parameters[1]
        if slope < 0:
            left_fit.append((slope,intercept))
        else:
            right_fit.append((slope,intercept))
    left_fit_average = np.average(left_fit , axis = 0)
    right_fit_average = np.average(right_fit , axis = 0)
    left_line = make_coordinates(image , left_fit_average)
    right_line = make_coordinates(image , right_fit_average)
    return np.array([left_line , right_line])

def display_lines(image , lines):
    line_image = np.zeros_like(image)
    if lines is not None:
        for x1 , y1 , x2 , y2 in lines:
            cv2.line(line_image , (x1 , y1) , (x2 , y2) , (0,0,255) , 5 )
    return line_image

cap = cv2.VideoCapture("Lane.mp4")
while(cap.isOpened()):
    _ , frame = cap.read()
    cany = canny(frame)
    cropped_image = region_of_intrests(cany)
    lines = cv2.HoughLinesP(cropped_image, 2 , np.pi/180 , 100 , np.array([]) , minLineLength = 40 , maxLineGap = 5 )
    averaged_lines = average_slpoe_intercept(frame,lines)
    line_image = display_lines(frame,averaged_lines)
    combo_image = cv2.addWeighted(frame , 0.8 , line_image , 1 , 1)
    cv2.imshow("result",combo_image)
    if cv2.waitKey(1) == ord('q'):
        break
cap.release()
cv2.destroyAllWindows()
