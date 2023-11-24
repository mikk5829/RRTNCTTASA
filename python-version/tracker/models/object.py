import matplotlib.pyplot as plt
import numpy as np

from models.moments import Coordinates, Moments, SimpleCoordinates
import cv2 as cv

from models.pose import Pose


def dynamic_threshold(blur_image, top_left, bottom_right, background_threshold, acc_img):
    cropped_img = blur_image[top_left[1]:bottom_right[1], top_left[0]:bottom_right[0]]
    ret, thresh_img = cv.threshold(cropped_img, 0, 255, cv.THRESH_BINARY +
                                   cv.THRESH_TRIANGLE)

    plt.imshow(thresh_img, cmap='gray')
    plt.show()

    # insert thresh_img into acc_img
    acc_img[top_left[1]:bottom_right[1], top_left[0]:bottom_right[0]] = thresh_img
    # find upper row of pixels that are above the background threshold
    upper_row = np.where(thresh_img[0, :] > background_threshold)
    bottom_row = np.where(thresh_img[-1, :] > background_threshold)
    left_column = np.where(thresh_img[:, 0] > background_threshold)
    right_column = np.where(thresh_img[:, -1] > background_threshold)

    # percent_search = 0.1
    # # if 50% of the pixels are above the threshold, we search in 10% above the top
    # if len(upper_row[0]) > thresh_img.shape[1] / 2:
    #     recursive_threshold(blur_image, (int(top_left[0]), int(top_left[1] - top_left[1] * percent_search)),
    #                         (bottom_right[0], top_left[1]), background_threshold, acc_img)
    # # if 50% of the pixels are above the threshold, we search in 10% below the bottom
    # if len(bottom_row[0]) > thresh_img.shape[1] / 2:
    #     recursive_threshold(blur_image, (top_left[0], bottom_right[1]),
    #                         (bottom_right[0], int(bottom_right[1] + bottom_right[1] * percent_search)),
    #                         background_threshold, acc_img)
    # # if 50% of the pixels are above the threshold, we search in 10% to the left
    # if len(left_column[0]) > thresh_img.shape[0] / 2:
    #     recursive_threshold(blur_image, (int(top_left[0] - top_left[0] * percent_search), top_left[1]),
    #                         (top_left[0], bottom_right[1]), background_threshold, acc_img)
    # # if 50% of the pixels are above the threshold, we search in 10% to the right
    # if len(right_column[0]) > thresh_img.shape[0] / 2:
    #     recursive_threshold(blur_image, (bottom_right[0], top_left[1]),
    #                         (int(bottom_right[0] + bottom_right[0] * percent_search), bottom_right[1]),
    #                         background_threshold, acc_img)

    return acc_img


class Corners:
    __top_left: SimpleCoordinates = None
    __top_right: SimpleCoordinates = None
    __bottom_left: SimpleCoordinates = None
    __bottom_right: SimpleCoordinates = None

    def __init__(self, top_left: SimpleCoordinates, top_right: SimpleCoordinates, bottom_left: SimpleCoordinates,
                 bottom_right: SimpleCoordinates):
        self.__top_left = top_left
        self.__top_right = top_right
        self.__bottom_left = bottom_left
        self.__bottom_right = bottom_right


class Object:
    """
    This class represents an object in an image
    """
    __moments: Moments = None
    __relative_contour: cv.UMat = None
    __aligned_image: cv.UMat = None
    __corners: Corners = None
    __bounding_box = None
    __raw_image: cv.UMat = None
    __pose: Pose = None
    __threshold_image: cv.UMat = None
    __contour: cv.UMat = None

    def __init__(self, raw_image, simplify_contours=False):
        self.__raw_image = raw_image
        self.__preprocess_image()
        self.__detect_contours(simplify_contours)
        self.__set_bounding_box()

    def __preprocess_image(self):
        # make a gaussian blur of the image to remove noise
        blur_image = cv.GaussianBlur(self.__raw_image, (5, 5), 0)

        # make all pixels in blur_image exponentially brighter
        # blur_image = np.power(blur_image, 1.5).astype(np.uint8)

        # hist = cv.calcHist([blur_image], [0], None, [256], [0, 256])
        # # plot log hist
        # plt.plot(hist)
        # plt.show()
        # dynamic_threshold = np.argmax(hist)  # below this is background

        # get x pixels evenly distributed from the image
        x = np.linspace(0, blur_image.shape[1] - 1, 20, dtype=int)
        y = np.linspace(0, blur_image.shape[0] - 1, 20, dtype=int)
        x, y = np.meshgrid(x, y)
        x = x.flatten()
        y = y.flatten()
        # get the pixel values at the coordinates
        pixel_values = blur_image[y, x]
        # get the histogram of the pixel values
        hist = cv.calcHist([pixel_values], [0], None, [256], [0, 256])
        # find the standard deviation of the histogram index
        std = np.std(pixel_values)  # - np.mean(pixel_values)
        plt.plot(hist)
        # plot std
        # plt.plot([std + np.mean(pixel_values), std + np.mean(pixel_values)], [0, np.max(hist)], "r")
        plt.show()

        background_threshold = np.argmax(hist) + std * 2

        # find the index of pixel_values that are above the background threshold in x and y
        above_background = np.where(pixel_values > background_threshold)
        # get the coordinates of the pixels that are above the background threshold
        x = x[above_background]
        y = y[above_background]

        # plot the coordinates
        plt.plot(x, y, "r*")
        plt.imshow(blur_image, cmap='gray')
        plt.show()

        # check different sobel ddepth
        sobelx = cv.Sobel(blur_image, cv.CV_16S, 1, 0, ksize=3)
        sobely = cv.Sobel(blur_image, cv.CV_16S, 0, 1, ksize=3)
        laplacian = cv.Laplacian(blur_image, cv.CV_16S, ksize=3)
        abs_grad_x = cv.convertScaleAbs(sobelx)
        abs_grad_y = cv.convertScaleAbs(sobely)

        grad = cv.addWeighted(abs_grad_x, 0.5, abs_grad_y, 0.5, 0)
        plt.imshow(sobelx, cmap='gray')
        plt.show()
        plt.imshow(sobely, cmap='gray')
        plt.show()
        plt.imshow(laplacian, cmap='gray')
        plt.show()
        abs_grad_laplacian = cv.convertScaleAbs(laplacian)
        plt.imshow(grad, cmap='gray')
        plt.show()

        # find contours in the image grad
        ret, self.__threshold_image = cv.threshold(grad, 2, 255, cv.THRESH_BINARY)
        contours, hierarchy = cv.findContours(self.__threshold_image, cv.RETR_TREE, cv.CHAIN_APPROX_SIMPLE)
        # find the longest contour
        contour = max(contours, key=cv.contourArea)
        # plot the contour
        plt.imshow(grad, cmap='gray')
        plt.plot(contour.squeeze()[:, 0], contour.squeeze()[:, 1], "r")
        plt.show()

        # make a sqaure around top left, top right, bottom left and bottom right
        top_left = (x.min(), y.min())
        bottom_right = (x.max(), y.max())

        # make a recursive thresholding of the image
        # acc_img = np.zeros((blur_image.shape[0], blur_image.shape[1]))
        # thresh_img = recursive_threshold(blur_image, top_left, bottom_right, background_threshold, acc_img)
        #
        # plt.imshow(thresh_img, cmap='gray')
        # plt.show()

        # inside each square we make otsu thresholding
        # we do the same for up, down, left and right
        # if less than 50% is above the threshold, we the square split into 4 squares

        # max_diff_index = None
        # for point in zip(x, y):
        #     # find the circle around the point
        #     circle_x = [0, 1, 2, 3, 3, 3, 2, 1, 0, -1, -2, -3, -3, -3, -2, -1]
        #     circle_y = [3, 3, 2, 1, 0, -1, -2, -3, -3, -3, -2, -1, 0, 1, 2, 3]
        #     # get the pixel values of the circle
        #     circle_values = blur_image[point[1] + circle_y, point[0] + circle_x]
        #     # get value of point
        #     point_value = blur_image[point[1], point[0]]
        #     # find the difference between the point value and the circle values
        #     diff = circle_values - point_value
        #     # find the index where difference is the highest
        #     max_diff_index = np.argmax(diff)
        #
        # # plot max_diff_index on the image
        # plt.imshow(blur_image, cmap='gray')
        # plt.plot(x[max_diff_index], y[max_diff_index], "r*")
        # plt.show()

        # select a random pixel from the above background pixels
        # random_index = np.random.randint(0, len(x))
        # Select a circle around the pixel

        # new_img = np.zeros((blur_image.shape[0], blur_image.shape[1]))
        # for i in range(1, 15, 5):
        #     new_img += self.__get_threshold_image_split(i, blur_image)
        #
        # ret, new_img = cv.threshold(new_img, 0, 255, cv.THRESH_BINARY + cv.THRESH_TRIANGLE)
        # plt.imshow(new_img, cmap='gray')
        # plt.show()

        # make a threshold of the blurred image to get a binary image
        # ret, self.__threshold_image = cv.threshold(blur_image, 0, 255, cv.THRESH_BINARY + cv.THRESH_TRIANGLE)
        # ret, self.__threshold_image = cv.threshold(new_img, 21, 255, cv.THRESH_BINARY)
        # ret_otsu, _ = cv.threshold(blur_image, 0, 255, cv.THRESH_BINARY + cv.THRESH_OTSU)
        # canny edge detection
        # median_pix = np.median(self.__threshold_image)
        # lower = int(max(0, 0.6 * median_pix))
        # upper = int(min(255, 1.3 * median_pix))
        # canny = cv.Canny(image=self.__threshold_image, threshold1=ret_otsu * 0.5, threshold2=ret_otsu)
        # canny = cv.Canny(blur_image, 0, 255)
        # plt.imshow(canny, cmap='gray')
        # plt.show()
        # plot self.__threshold_image
        # plt.imshow(self.__threshold_image, cmap='gray')
        # plt.show()
        # coordinates = self.__moments.coordinates()
        # self.__threshold_image = rotate_image_around_center_of_mass(self.__threshold_image,
        #                                                             coordinates.orientation_degrees,
        #                                                             coordinates.x, coordinates.y)

    def __simplify_contours(self):
        """
        This function is used to simplify the contours of the object
        """
        epsilon = 0.0025 * cv.arcLength(self.__contour, True)
        approx = cv.approxPolyDP(self.__contour, epsilon, True)
        self.__contour = approx

    def __get_threshold_image_split(self, n, blur_image):
        assert n > 0
        # todo start at region of interest
        # find dynamically the threshold value using a histogram and the 1st peak
        hist = cv.calcHist([blur_image], [0], None, [256], [0, 256])
        dynamic_threshold = np.argmax(hist)

        width = int(blur_image.shape[1] / n)
        height = int(blur_image.shape[0] / n)
        new_img = np.zeros((blur_image.shape[0], blur_image.shape[1]))
        for i in range(n):
            for j in range(n):
                x = (1 + i) * width
                y = (1 + j) * height
                cropped_img = blur_image[y:y + height, x:x + width]
                ret, thresh_img = cv.threshold(cropped_img, 0, 255, cv.THRESH_BINARY +
                                               cv.THRESH_TRIANGLE)
                if ret > dynamic_threshold:
                    cropped_img = thresh_img

                new_img[y:y + height, x:x + width] = cropped_img

        return new_img

    def __detect_contours(self, simplify_contours):
        contours, hierarchy = cv.findContours(self.__threshold_image, cv.RETR_TREE, cv.CHAIN_APPROX_SIMPLE)

        # find the longest contour
        self.__contour = max(contours, key=cv.contourArea)

        if simplify_contours:
            self.__simplify_contours()

        # find bounding box coordinates and crop the image
        img_x, img_y, img_width, img_height = cv.boundingRect(self.__contour)
        self.__threshold_image = self.__threshold_image[img_y:img_y + img_height, img_x:img_x + img_width]

        self.__threshold_image = cv.resize(self.__threshold_image, (800, 800), interpolation=cv.INTER_AREA)

        self.__moments = Moments(self.__contour)

        coordinates = self.__moments.get_coordinates()
        # minus the center of mass from the contour to get the relative contour
        relative_cnt = self.__contour - coordinates.as_tuple()

        self.__relative_contour = relative_cnt

    def __align_image(self):
        color_img = cv.cvtColor(self.__threshold_image, cv.COLOR_GRAY2RGB)
        coordinates = self.__moments.get_coordinates()
        cv.circle(color_img, (coordinates.x, coordinates.y), 7, (0, 0,
                                                                 255), -1)
        cv.drawContours(color_img, [self.__contour], 0, (0, 255, 0), 3)

        for corner in self.__corners.__dict__.values():
            cv.circle(color_img, (corner.x, corner.y), 5, (255, 0, 0), 3)

        # draw bounding box
        cv.drawContours(color_img, self.__bounding_box, 0, (0, 0, 255), 2)

        self.__aligned_image = color_img

    def get_aligned_image(self) -> cv.UMat or None:
        if self.__aligned_image is None:
            self.__align_image()
        return self.__aligned_image

    def get_raw_image(self) -> cv.UMat or None:
        return self.__raw_image

    def __set_bounding_box(self):
        ct = self.__contour.squeeze()
        # find the top left corner
        bottom_left = SimpleCoordinates(ct[ct[:, 0].argmin()])
        top_right = SimpleCoordinates(ct[ct[:, 0].argmax()])
        top_left = SimpleCoordinates(ct[ct[:, 1].argmin()])
        bottom_right = SimpleCoordinates(ct[ct[:, 1].argmax()])

        self.__corners = Corners(top_left, top_right, bottom_left, bottom_right)

        # find the bounding box
        min_area_rect = cv.minAreaRect(self.__contour)
        bounding_box = cv.boxPoints(min_area_rect)
        bounding_box = bounding_box.astype('int')

        self.__bounding_box = [bounding_box]

    def get_relative_contour(self) -> cv.UMat:
        return self.__relative_contour

    def get_contour(self) -> cv.UMat:
        return self.__contour

    def get_moments(self) -> Moments:
        return self.__moments

    @property
    def coordinates(self) -> Coordinates:
        return self.__moments.get_coordinates()

    def __str__(self) -> str:
        return f"Coordinates: {self.__moments.get_coordinates()}"


def rotate_image_around_center_of_mass(image: cv.UMat, angle_in_degrees, center_x, center_y):
    rotation_matrix = cv.getRotationMatrix2D((center_x, center_y), angle_in_degrees, 1.0)
    result = cv.warpAffine(image, rotation_matrix, image.shape[1::-1], flags=cv.INTER_LINEAR)
    return result
