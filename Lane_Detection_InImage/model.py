import cv2
import numpy as np
import matplotlib.pyplot as plt

def canny(lane_image):   #function to convert an image into edge detected image(image with edges)
    gray = cv2.cvtColor(lane_image,cv2.COLOR_RGB2GRAY)   #Converts normal image to grey image(almost black and white)
    blur = cv2.GaussianBlur(gray,(5,5),0)    # Makes the image smooth 
    cany = cv2.Canny(blur,50,120)  # Detects the edged and converts edges to white and the whole other image to black
    return cany

def region_of_intrests(image):      # returns an image with a shape of desired polygon
    polygon = np.array([[(0,382),(0,95),(200,0),(485,0),(765,145),(765,382)]])  # shape of polygon
    mask = np.zeros_like(image)     # Creates a black image with same dimensions as desired image 
    cv2.fillPoly(mask,polygon,255)   # Creates a white region in the mask with the shape of polygon
    masked_image = cv2.bitwise_and(image,mask)  # Copys the polygon shape of image into mask
    return masked_image


'''
///////        This commented part is used to convert all the lines which are near to each other into one line by obtaining their average        ///////

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

'''

def display_lines(image , lines):
    line_image = np.zeros_like(image)
    if lines is not None:
        for line in lines:
            x1 , y1 , x2 , y2 = line.reshape(4)
            cv2.line(line_image , (x1 , y1) , (x2 , y2) , (0,255,0) , 5 )
    return line_image

image = cv2.imread('/Users/gamemaster/Downloads/Lane1.jpg')
lane_image = np.copy(image)
cany = canny(lane_image)
cropped_image = region_of_intrests(cany)
lines = cv2.HoughLinesP(cropped_image, 2 , np.pi/180 , 15 , np.array([]) , minLineLength = 1 , maxLineGap = 5 )


# averaged_lines = average_slpoe_intercept(lane_image,lines)  # If u need to use this line then replace "lines" in "line_image" with "averaged_lines" .


line_image = display_lines(lane_image,lines)
combo_image = cv2.addWeighted(lane_image , 0.8 , line_image , 1 , 1)
plt.imshow(combo_image)
plt.show()
