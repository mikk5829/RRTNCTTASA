from models.moments import Moments, SimpleCoordinates
import cv2 as cv

from models.pose import Pose


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

        # make a threshold of the blurred image to get a binary image
        # ret, self.__threshold_image = cv.threshold(blur_image, 0, 255, cv.THRESH_BINARY + cv.THRESH_TRIANGLE)
        ret, self.__threshold_image = cv.threshold(blur_image, 25, 255, cv.THRESH_BINARY)

        self.__moments = Moments(self.__threshold_image)

        coordinates = self.__moments.get_coordinates()
        self.__threshold_image = rotate_image_around_center_of_mass(self.__threshold_image,
                                                                    coordinates.orientation_degrees,
                                                                    coordinates.x, coordinates.y)

    def __simplify_contours(self):
        """
        This function is used to simplify the contours of the object
        """
        epsilon = 0.002 * cv.arcLength(self.__contour, True)
        approx = cv.approxPolyDP(self.__contour, epsilon, True)
        self.__contour = approx

    def __detect_contours(self, simplify_contours):
        contours, hierarchy = cv.findContours(self.__threshold_image, cv.RETR_TREE, cv.CHAIN_APPROX_SIMPLE)

        # find the longest contour
        self.__contour = max(contours, key=cv.contourArea)

        if simplify_contours:
            self.__simplify_contours()

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

    def __str__(self) -> str:
        return f"Coordinates: {self.__moments.get_coordinates()}"


def rotate_image_around_center_of_mass(image: cv.UMat, angle_in_degrees, center_x, center_y):
    rotation_matrix = cv.getRotationMatrix2D((center_x, center_y), angle_in_degrees, 1.0)
    result = cv.warpAffine(image, rotation_matrix, image.shape[1::-1], flags=cv.INTER_LINEAR)
    return result
