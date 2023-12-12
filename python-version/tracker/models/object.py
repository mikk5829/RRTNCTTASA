import matplotlib.pyplot as plt
import numpy as np

from models.moments import Coordinates, Moments, SimpleCoordinates
import cv2 as cv

from models.pose import Pose, Rotation, Translation


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
    __rotation: Rotation = None
    __threshold_image: cv.UMat = None
    __contour: cv.UMat = None
    __verbose = False
    __satellite_points = None

    def __init__(self, raw_image, simplify_contours=False, verbose=False, is_model=False):
        self.__verbose = verbose
        self.__raw_image = raw_image
        self.__preprocess_image(is_model)
        self.__detect_contours(simplify_contours)
        self.__set_bounding_box()

    def get_rotation(self):
        return self.__rotation

    def set_rotation(self, rotation: Rotation):
        self.__rotation = rotation

    def set_roll(self, roll):
        self.__pose.set_roll(roll)

    def __preprocess_image(self, is_model=False):
        # make a gaussian blur of the image to remove noise
        blur_image = cv.GaussianBlur(self.__raw_image, (5, 5), 0)

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
        if self.__verbose:
            plt.plot(hist)
            plt.show()

        background_threshold = np.argmax(hist) + std * 2

        # find the index of pixel_values that are above the background threshold in x and y
        above_background = np.where(pixel_values > background_threshold)
        # get the coordinates of the pixels that are above the background threshold
        x = x[above_background]
        y = y[above_background]

        self.__satellite_points = np.array([x, y]).T

        # plot the coordinates
        if self.__verbose:
            plt.plot(x, y, "r*")
            plt.imshow(blur_image, cmap='gray')
            plt.show()

        if not is_model:
            # check different sobel ddepth
            sobelx = cv.Sobel(blur_image, cv.CV_16S, 1, 0, ksize=3)
            sobely = cv.Sobel(blur_image, cv.CV_16S, 0, 1, ksize=3)
            abs_grad_x = cv.convertScaleAbs(sobelx)
            abs_grad_y = cv.convertScaleAbs(sobely)

            grad = cv.addWeighted(abs_grad_x, 0.5, abs_grad_y, 0.5, 0)
            # if self.__verbose:
            #     plt.imshow(sobelx, cmap='gray')
            #     plt.show()
            #     plt.imshow(sobely, cmap='gray')
            #     plt.show()
            #     plt.imshow(grad, cmap='gray')
            #     plt.show()

            # find contours in the image grad
            ret, self.__threshold_image = cv.threshold(grad, 2, 255, cv.THRESH_BINARY)
        else:
            ret, self.__threshold_image = cv.threshold(blur_image, 0, 255, cv.THRESH_BINARY + cv.THRESH_TRIANGLE)

        # contours, hierarchy = cv.findContours(self.__threshold_image, cv.RETR_TREE, cv.CHAIN_APPROX_SIMPLE)
        # # find the longest contour
        # contour = max(contours, key=cv.contourArea)
        # self.__contour = contour
        # plot the contour
        # if self.__verbose:
        #     plt.imshow(grad, cmap='gray')
        #     plt.plot(contour.squeeze()[:, 0], contour.squeeze()[:, 1], "r")
        #     plt.show()

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
        contours, hierarchy = cv.findContours(self.__threshold_image, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)

        # remove the contours that are too small
        contours = [contour for contour in contours if cv.contourArea(contour) > 50]

        # TODO remove contours that are not containing points from all the satellite points
        contours = [contour for contour in contours if
                    np.any(np.isin(self.__satellite_points, contour.squeeze(), True).all(axis=1))]

        # plot contours
        if self.__verbose:
            plt.imshow(self.__threshold_image, cmap='gray')
            # plt.plot(self.__contour.squeeze()[:, 0], self.__contour.squeeze()[:, 1], "r")
            for contour in contours:
                plt.plot(contour.squeeze()[:, 0], contour.squeeze()[:, 1], "r")
            plt.show()

        self.__contour = max(contours, key=cv.contourArea)

        # if self.__verbose:
        #     plt.imshow(self.__threshold_image, cmap='gray')
        #     plt.plot(self.__contour.squeeze()[:, 0], self.__contour.squeeze()[:, 1], "r")
        #     plt.show()

        # cont = np.vstack([contours[i] for i in range(len(contours))])
        # hull = cv.convexHull(cont)
        #
        # self.__contour = hull

        if simplify_contours:
            self.__simplify_contours()

            if self.__verbose:
                plt.imshow(self.__threshold_image, cmap='gray')
            plt.plot(self.__contour.squeeze()[:, 0], self.__contour.squeeze()[:, 1], "r")
            plt.show()

        # find the n longest lines in self.__contour
        # n = 10
        # lines = []
        # for i in range(len(self.__contour)):
        #     lines.append(cv.norm(self.__contour[i] - self.__contour[i - 1]))
        # longest_lines = np.argsort(lines)[-n:]
        # # get the coordinates of the longest lines
        # longest_lines_coordinates = self.__contour.squeeze()[longest_lines]
        # # plot the longest lines on the contour
        # if self.__verbose:
        #     plt.plot(self.__contour.squeeze()[:, 0], self.__contour.squeeze()[:, 1], "r")
        #     plt.plot(longest_lines_coordinates[:, 0], longest_lines_coordinates[:, 1], "b")
        #     plt.show()

        # shift self.__contour by 1
        contour_local = self.__contour
        shifted_contour = np.roll(self.__contour, -1, axis=0)
        diff = self.__contour - shifted_contour
        lines = np.concatenate([contour_local, shifted_contour], axis=1)
        lengths = np.sqrt(np.sum(diff ** 2, axis=2))
        # find n longest lines
        n = 3
        longest_line_index = np.argsort(lengths.squeeze())[-n:]
        longest_line_index = sorted(longest_line_index)
        lines = lines[longest_line_index]

        # add 1 to longest_line_index to get the next point remember to wrap around
        longest_line_index_all = np.array(longest_line_index) + 1
        longest_line_index_all = np.mod(longest_line_index_all, len(self.__contour))

        # select longest_line_index from self.__contour
        newLines = self.__contour.squeeze()[longest_line_index_all]
        # reshape newLines to  self.__contour.shape

        if self.__verbose:
            plt.plot(self.__contour.squeeze()[:, 0], self.__contour.squeeze()[:, 1], "r")
            # plot the lines (coordinates) on the contour
            for line in lines:
                plt.plot(line[:, 0], line[:, 1], "b")
            plt.show()

        # self.__contour = newLines.reshape((newLines.shape[0], 1, newLines.shape[1]))

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

    def set_pose(self, pose):
        self.__pose = pose


def rotate_image_around_center_of_mass(image: cv.UMat, angle_in_degrees, center_x, center_y):
    rotation_matrix = cv.getRotationMatrix2D((center_x, center_y), angle_in_degrees, 1.0)
    result = cv.warpAffine(image, rotation_matrix, image.shape[1::-1], flags=cv.INTER_LINEAR)
    return result
