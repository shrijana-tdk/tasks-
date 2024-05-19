import cv2
import numpy as np
import matplotlib.pyplot as plt

img= cv2.imread('img.png',cv2.IMREAD_GRAYSCALE)

#imgray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
ret, thresh = cv2.threshold(img, 127, 255, 0)
contours, hierarchy = cv2.findContours(thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
print(len(contours))

# To create a black image of the same size as the original image to draw contours
contour_image = np.ones_like(img) * 255
# Draw the contour at index (1-12) 
# cv2.drawContours(contour_image, contours, 13, 255, thickness=cv2.FILLED)

contours_to_rotate=[1,3,4,6,7,9,10,12]

def cart2pol(x, y):
    theta = np.arctan2(y, x)
    rho = np.hypot(x, y)
    return theta, rho

def pol2cart(theta, rho):
    x = rho * np.cos(theta)
    y = rho * np.sin(theta)
    return x, y

def rotate_contour(cnt, angle):
    M = cv2.moments(cnt)
    # cx, cy are centroid of the contour cnt which are calculated using its moments. 
    # The centroid (cx, cy) is the point around which the contour will be rotated.
    cx = int(M['m10'] / M['m00']) 
    cy = int(M['m01'] / M['m00'])

    cnt_norm = cnt - [cx, cy]  # subtracting the centroid coordinates from all contour points.
    # Translate the contour so that its centroid is at the origin (0, 0). 

    coordinates = cnt_norm[:, 0, :]
    xs, ys = coordinates[:, 0], coordinates[:, 1]
    thetas, rhos = cart2pol(xs, ys)
    # Convert the normalized contour coordinates to polar coordinates (thetas, rhos). 
    # This step is done using the cart2pol function, which converts Cartesian coordinates (x, y) to polar coordinates (theta, rho).

    thetas = np.rad2deg(thetas)
    thetas = (thetas + angle) % 360  # Adding the specified angle to the theta values (angles) to rotate the contour. 
    thetas = np.deg2rad(thetas)
    # The angles are in the range [0, 360) as taking the modulus 360. 
    # Then converting the angles back to radians.
    
    xs, ys = pol2cart(thetas, rhos) # The polar coordinates are converted back to Cartesian coordinates (xs, ys)

    cnt_norm[:, 0, 0] = xs
    cnt_norm[:, 0, 1] = ys

    cnt_rotated = cnt_norm + [cx, cy] # The centroid coordinates is added back to the rotated contour coordinates to restore its original position.
    cnt_rotated = cnt_rotated.astype(np.int32)

    return cnt_rotated

for i in contours_to_rotate:
    rect = cv2.minAreaRect(contours[i])
    box = cv2.boxPoints(rect)
    box = np.int0(box)

    if box[0][1] < box[1][1]:
        angle = rect[-1]
    else:
        angle =90 + rect[-1]

    angle = -angle  # Adjusting angle for correct rotation
    cnt_rotated = rotate_contour(contours[i], angle)
    cv2.drawContours(contour_image, [cnt_rotated], 0, (0, 0, 0), 4)


    # Find the orientation of the bounding rectangle
    #angle = cv2.minAreaRect(contours[i])[-1]
#    angle = rect[-1]
#    if angle < -45:
#        angle = -(90 + angle)  
#    else:
#       angle = -angle 
#    cnt_rotated = rotate_contour(contours[i], angle)
#    cv2.drawContours(contour_image, [cnt_rotated], -1, 255, thickness= 2)


cv2.imshow("output",contour_image)
cv2.imwrite("op2.png",contour_image)
cv2.waitKey(0)
