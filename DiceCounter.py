import cv2
import numpy as np;


BadAnswer = True

# Read image
im = cv2.imread("Images/dices4.jpg", cv2.IMREAD_GRAYSCALE)
imr = cv2.resize(im, (640, 480))
(thresh, imbw) = cv2.threshold(im, 150, 255, cv2.THRESH_BINARY)

# Setup SimpleBlobDetector parameters.
params = cv2.SimpleBlobDetector_Params()


# while BadAnswer:
#     color = input("Fehérek(w) vagy feketék(b) a pontok a dobókockán:  ")
#     if color == "w":
#         params.blobColor = 1;  
#         BadAnswer = False  
#     if color == "b":
#         params.blobColor = 0;
#         BadAnswer = False 

params.filterByColor = True

# Change thresholds
params.minThreshold = 20;
params.maxThreshold = 200;
 
# Filter by Area.
params.filterByArea = True
params.minArea = 50
params.maxArea = 1500
 
# Filter by Circularity
params.filterByCircularity = True
params.minCircularity = 0.77
params.maxCircularity = 1.0

# Filter by Convexity
params.filterByConvexity = True
params.minConvexity = 0.8
 
# Filter by Inertia
params.filterByInertia = True
params.minInertiaRatio = 0.01


# Set up the detector with default parameters.
detector = cv2.SimpleBlobDetector_create(params)

# Detect blobs.
keypoints = detector.detect(imr)

# Draw detected blobs as red circles.
# cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS ensures the size of the circle corresponds to the size of blob
im_with_keypoints = cv2.drawKeypoints(imr, keypoints, np.array([]), (0,0,255), cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)

# Show keypoints
cv2.imshow("Keypoints", im_with_keypoints)
cv2.waitKey(0)
