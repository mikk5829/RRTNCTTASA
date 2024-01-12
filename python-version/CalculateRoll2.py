import numpy as np
import matplotlib.pyplot as plt
import sys
import math
import cv2 as cv


def main(argv):
    img_name = "..blender\\0.363_-0.759_0.541.png"
    # read image as grayscale image
    img = cv.imread(img_name, cv.IMREAD_GRAYSCALE)
    # gaussian blur the image
    img = cv.GaussianBlur(img, (5, 5), 0)

    img_name2 = "..blender\\0.369_-0.548_0.751.png"
    # read image as grayscale image
    img2 = cv.imread(img_name2, cv.IMREAD_GRAYSCALE)
    # gaussian blur the image
    img2 = cv.GaussianBlur(img2, (5, 5), 0)

    # use  triangle algorithm to find the threshold
    ret, thresh = cv.threshold(img, 0, 255, cv.THRESH_BINARY + cv.THRESH_TRIANGLE)
    _, thresh2 = cv.threshold(img2, 0, 255, cv.THRESH_BINARY + cv.THRESH_TRIANGLE)

    # print the triangle algorithm threshold value
    print("Threshold value is {}".format(ret))

    contours, _ = cv.findContours(thresh, cv.RETR_TREE, cv.CHAIN_APPROX_SIMPLE)
    cnt = max(contours, key=cv.contourArea)

    contours2, _ = cv.findContours(thresh2, cv.RETR_TREE, cv.CHAIN_APPROX_SIMPLE)
    cnt2 = max(contours2, key=cv.contourArea)

    # draw contours
    img = cv.cvtColor(img, cv.COLOR_GRAY2RGB)
    img = cv.drawContours(img, [cnt], -1, (0, 0, 255), 3)

    # Find the CoM of the contour
    M = cv.moments(cnt)
    cx = int(M['m10'] / M['m00'])
    cy = int(M['m01'] / M['m00'])
    # Find the area of the contour
    area = cv.contourArea(cnt)

    M2 = cv.moments(cnt2)
    cx2 = int(M2['m10'] / M2['m00'])
    cy2 = int(M2['m01'] / M2['m00'])
    # Find the area of the contour
    area2 = cv.contourArea(cnt2)

    print("Scale factor from contour area: ", area2 / area)

    fs = area2 / area
    CoM = [cx2, cy2]
    height, width, _ = img.shape
    print("XYZ: ", XYZ(width, height, CoM, fs))

    # Plot the CoM on the image
    cv.circle(img, (cx, cy), 5, (255, 0, 0), -1)
    plt.imshow(img, cmap='gray')
    plt.show()

    # Calculate the distance between every point and the CoM
    dist = np.sqrt((cnt[:, :, 0] - cx) ** 2 + (cnt[:, :, 1] - cy) ** 2)
    # Calculate the angle between every point and the CoM
    angle = np.arctan2(cnt[:, :, 1] - cy, cnt[:, :, 0] - cx)
    # convert angles to degrees
    angle = np.degrees(angle)
    angle = np.mod(angle, 360)

    # Plot the distance and angle with angles from 0 to 360
    # sort angles from 0 to 360 and also the corresponding distances
    angle, dist = zip(*sorted(zip(angle, dist)))
    # plot the distance as a function of angle

    # Calculate the distance between every point and the CoM
    dist_shift = np.sqrt((cnt2[:, :, 0] - cx2) ** 2 + (cnt2[:, :, 1] - cy2) ** 2)
    # Calculate the angle between every point and the CoM
    angle_shift = np.arctan2(cnt2[:, :, 1] - cy2, cnt2[:, :, 0] - cx2)
    # convert angles to degrees
    angle_shift = np.degrees(angle_shift)
    angle_shift = np.mod(angle_shift, 360)

    angle_shift, dist_shift = zip(*sorted(zip(angle_shift, dist_shift)))

    dist = np.asarray(dist).squeeze()
    angle = np.asarray(angle).squeeze()

    # do the same for angle_shift and dist_shift
    dist_shift = np.asarray(dist_shift).squeeze()
    angle_shift = np.asarray(angle_shift).squeeze()

    plt.plot(angle, dist)
    plt.plot(angle_shift, dist_shift)
    plt.show()

    rollAngle = calculate_roll2(angle, dist, angle_shift, dist_shift)
    rollAngle2 = (angle_shift[np.where(dist_shift == max(dist_shift))] - angle[np.where(dist == max(dist))])[0] % 360

    print("Roll angle - circular cross-correlation (deg): ", rollAngle, u"\u00B0")
    print("Roll angle - peak2peak (deg): ", rollAngle2, u"\u00B0")


def calculate_roll2(A1, A2, B1, B2):
    parametric_curve = custom_uniform_interpolation(A1, A2)
    parametric_curve2 = custom_uniform_interpolation(B1, B2)

    mult_sum = np.zeros(parametric_curve.shape[0])
    for i in range(len(mult_sum)):
        mult_sum[i] = np.dot(parametric_curve, np.roll(parametric_curve2, -i))

    # Find the index of the maximum value.
    max_idx = np.argmax(mult_sum)
    # draw a vertical line at the maximum value.
    plt.axvline(max_idx, color='r')

    plt.plot(mult_sum)
    plt.show()

    plt.plot(A1, A2, 'o', label='Original Signal')
    plt.plot(B1, B2, 'o', label='Original Signal')
    plt.plot(np.arange(360), parametric_curve, label='Interpolated Signal')
    plt.plot(np.arange(360), parametric_curve2, label='Interpolated Signal')
    plt.legend()
    plt.show()

    return max_idx


def custom_uniform_interpolation(X, Y):
    # Custom uniform interpolation to make X uniformly distributed between 0 and 360 degrees with spacing of 1 degree.

    # Ensure that X is within [0, 360] and sorted in ascending order.
    X = np.mod(X, 360)
    sorted_idx = np.argsort(X)
    X = X[sorted_idx]
    Y = Y[sorted_idx]

    # Prepare the data for interpolation.
    a = (Y[sorted_idx[0]] - Y[sorted_idx[-1]]) / abs(X[sorted_idx[-1]] - X[sorted_idx[0]] - 360)

    XY_end = [359, Y[sorted_idx[-1]] + a * (359 - X[sorted_idx[-1]])]
    XY_1 = [0, Y[sorted_idx[0]] + a * (0 - X[sorted_idx[0]])]

    X = np.concatenate(([XY_1[0]], X, [XY_end[0]]))
    Y = np.concatenate(([XY_1[1]], Y, [XY_end[1]]))

    # Initialize interpolated signal arrays.
    interpolated_X = np.arange(360)
    interpolated_Y = np.zeros(360)

    # Perform custom linear interpolation for each degree.
    for i, current_degree in enumerate(interpolated_X):
        # Find the closest points in X for interpolation.
        # print(np.size(X))
        try:
            lower_idx = np.where(X <= current_degree)[0][-1]
        except:
            lower_idx = None
        try:
            upper_idx = np.where(X > current_degree)[0][0]
        except:
            upper_idx = None

        if lower_idx is None or upper_idx is None:
            # If outside the original range, use the nearest endpoint.
            if current_degree <= X[0]:
                lower_idx = 0
                upper_idx = 1
            else:
                lower_idx = len(X) - 2
                upper_idx = len(X) - 1

        # Interpolation weights.
        alpha = (current_degree - X[lower_idx]) / (X[upper_idx] - X[lower_idx])

        # Custom linear interpolation.
        interpolated_Y[i] = (1 - alpha) * Y[lower_idx] + alpha * Y[upper_idx]

    return interpolated_Y


def XYZ(width, height, CoM, fs):
    m = 1
    mm = 1e-3
    um = 1e-6

    # An Effective Focal Length of:
    f = 20 * mm

    # and a pixel size on the CCD of:
    CCDy = 8.6 * um

    # What is the distance from the camera to the model in Blender?
    zm = 10 * m

    # The bigger the scale factor, the closer it is to us
    z = zm / fs

    cx = math.floor(width / 2)
    cy = math.floor(height / 2)
    px = ((CoM[0] - cx) * CCDy * z) / f
    py = ((CoM[1] - cy) * CCDy * z) / f

    XYZ = [px, py, z]

    return XYZ


if __name__ == "__main__":
    main(sys.argv[1:])