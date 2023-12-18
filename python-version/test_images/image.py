import cv2 as cv

alpha = 1.0
beta = 0


def set_alpha(new_alpha):
    global alpha
    alpha = new_alpha
    on_trackbar_change()


def set_beta(new_beta):
    global beta
    beta = new_beta
    on_trackbar_change()


def on_trackbar_change():
    # change brightness
    im = cv.imread("setup 5/A231212_10145093.png");
    im = cv.convertScaleAbs(im, alpha=alpha, beta=beta)

    # show image
    cv.imshow("Image", im)


# create trackbar
cv.namedWindow("Image", cv.WINDOW_NORMAL)
cv.createTrackbar("Brightness", "Image", 0, 100, lambda val: set_beta(val))
cv.createTrackbar("Contrast", "Image", 1, 100, lambda val: set_alpha(val))

on_trackbar_change()
cv.waitKey(0)
