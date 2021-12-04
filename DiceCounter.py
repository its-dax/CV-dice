import cv2
import numpy as np

impath = "Images/dices20.jpg"

# Read image
im = cv2.imread(impath)
img = cv2.cvtColor(im, cv2.COLOR_BGR2GRAY)
imr = cv2.resize(img, (640, 480))
imb = cv2.GaussianBlur(imr, (11,11), 5)
dilated = cv2.dilate(imb, (2,2), iterations = 15)
canny = cv2.Canny(dilated, 70, 60, 130)

dilated2 = cv2.dilate(canny, (1,1), iterations = 4)


# Setup SimpleBlobDetector parameters.
params = cv2.SimpleBlobDetector_Params()
params.filterByColor = True
params.minThreshold = 20;
params.maxThreshold = 200;
params.filterByArea = True
params.minArea = 77
params.maxArea = 700
params.filterByCircularity = True
params.minCircularity = 0.75
params.maxCircularity = 1.0
params.filterByConvexity = True
params.minConvexity = 0.70
params.filterByInertia = True
params.minInertiaRatio = 0.01

# Set up the detector with default parameters.
detector = cv2.SimpleBlobDetector_create(params)
# Detect blobs.
keypoints = detector.detect(imr)
#Number of keypoints detected
value = len(keypoints)

# Draw detected blobs as red circles.
im_with_keypoints = cv2.drawKeypoints(imr, keypoints, np.array([]), (0,0,255), cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)
#Put value on image 
cv2.putText(im_with_keypoints,f"Value: {value} ", (10,30), cv2.FONT_HERSHEY_COMPLEX, 0.75, (0,0,255),2)



                                ##############################################
                                ###############Counting dice##################
                                



# Setup SimpleBlobDetector parameters.
params2 = cv2.SimpleBlobDetector_Params()
params2.filterByColor = False
params2.filterByArea = True
params2.minArea = 5000
params2.maxArea = 18000
params2.filterByCircularity = False
params2.minCircularity = 0
params2.maxCircularity = 1.0
params2.filterByConvexity = False
params2.filterByInertia = False

# Set up the detector with default parameters.
detector2 = cv2.SimpleBlobDetector_create(params2)
# Detect blobs.
keypoints2 = detector2.detect(dilated)
#Number of keypoints detected
diceNum = len(keypoints2)

# Draw detected blobs as red circles.
im_with_keypoints = cv2.drawKeypoints(im_with_keypoints, keypoints2, np.array([]), (0,255,0), cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)




cv2.putText(im_with_keypoints,f"Number: {diceNum} ", (10,60), cv2.FONT_HERSHEY_COMPLEX, 0.75, (0,255,0),3)

# Show keypoints
cv2.imshow("Keypoints", im_with_keypoints)
#cv2.imshow("Keypoints2", im_with_keypoints2)

cv2.waitKey(0)
