import numpy as np
import cv2 as cv

# Read the query image as query_img
# and train image This query image
# is what you need to find in train image
# Save it in the same directory
# with the name image.jpg
query_img = cv.imread('0.363_-0.759_0.541.png')
train_img = cv.imread('0.369_-0.548_0.751.png')

# Convert it to grayscale
query_img_bw = cv.cvtColor(query_img, cv.COLOR_BGR2GRAY)
train_img_bw = cv.cvtColor(train_img, cv.COLOR_BGR2GRAY)

ret, query_img_thresh = cv.threshold(query_img_bw, 0, 255, cv.THRESH_BINARY + cv.THRESH_TRIANGLE)
ret, train_img_thresh = cv.threshold(train_img_bw, 0, 255, cv.THRESH_BINARY + cv.THRESH_TRIANGLE)

query_img_cnts, hierarchy = cv.findContours(query_img_thresh, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)
# query_img_cnts = query_img_cnts[0]
train_img_cnts, hierarchy = cv.findContours(train_img_thresh, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)
# train_img_cnts = train_img_cnts[0]
# find the longest contour
query_img_cnts = max(query_img_cnts, key=cv.contourArea)
train_img_cnts = max(train_img_cnts, key=cv.contourArea)
# find bounding box coordinates
query_img_x, query_img_y, query_img_w, query_img_h = cv.boundingRect(query_img_cnts)
train_img_x, train_img_y, train_img_w, train_img_h = cv.boundingRect(train_img_cnts)

# split query_img into channels
red_q, green_q, blue_q = cv.split(query_img)
red_t, green_t, blue_t = cv.split(train_img)

# select the channel
q_t = blue_q
t_t = blue_t

# crop the image
# q_t = q_t[320:945, 600:1250]
# t_t = t_t[320:945, 600:1250]

q_t = q_t[query_img_y:query_img_y + query_img_h, query_img_x:query_img_x + query_img_w]
t_t = t_t[train_img_y:train_img_y + train_img_h, train_img_x:train_img_x + train_img_w]

# Initialize the ORB detector algorithm
orb = cv.ORB_create()

# Now detect the keypoints and compute
# the descriptors for the query image
# and train image
queryKeypoints, queryDescriptors = orb.detectAndCompute(q_t, None)
trainKeypoints, trainDescriptors = orb.detectAndCompute(t_t, None)

# Initialize the Matcher for matching
# the keypoints and then match the
# keypoints
matcher = cv.BFMatcher(cv.NORM_HAMMING2, crossCheck=True)
matches = matcher.match(queryDescriptors, trainDescriptors)
matches = sorted(matches, key=lambda x: x.distance)

# draw the matches to the final image
# containing both the images the drawMatches()
# function takes both images and keypoints
# and outputs the matched query image with
# its train image
final_img = cv.drawMatches(q_t, queryKeypoints,
                           t_t, trainKeypoints, matches[:9], None,
                           flags=cv.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS)

final_img = cv.resize(final_img, (1000, 650))

# Show the final image
cv.imshow("Matches", final_img)
cv.waitKey()
