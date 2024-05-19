import cv2

image= cv2.imread('img.png')

img_gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
edges= cv2.Canny(img_gray, 50,200)
ret, threshold = cv2.threshold(img_gray, 150, 255, cv2.THRESH_BINARY)
contours, hierarchy = cv2.findContours(threshold, cv2.RETR_CCOMP, cv2.CHAIN_APPROX_NONE)

def get_contour_areas(contours):
    all_areas= []
    for c in contours:
        area= cv2.contourArea(c)
        all_areas.append(area)

    return all_areas

sorted_contours= sorted(contours, key=cv2.contourArea, reverse= False)
font = cv2.FONT_HERSHEY_DUPLEX
for i in range(0, min(4, len(sorted_contours))):
    M = cv2.moments(sorted_contours[i])
    cX = int(M["m10"] / M["m00"])
    cY = int(M["m01"] / M["m00"])
    cv2.putText(image, str(i+1), (cX-10, cY+88), font, 1, (255, 0, 0), 2, cv2.LINE_AA)


cv2.imshow("output",image)
cv2.imwrite("op1.png",image)
cv2.waitKey(0)