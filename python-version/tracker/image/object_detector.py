import cv2 as cv

from models.moments import Moments
from models.object import Object


class ObjectDetector:
    __tracking_object: Object = None
    __grey_scale_image = None
    __preprocessed_image = None
    aligned_image = None

    def __init__(self):
        # reset the state
        self.__tracking_object = None
        self.__grey_scale_image = None
        self.__preprocessed_image = None

    def __preprocess_image(self):
        # make a gaussian blur of the image to remove noise
        blur_image = cv.GaussianBlur(self.__grey_scale_image, (5, 5), 0)

        # make a threshold of the blurred image to get a binary image
        ret, self.__preprocessed_image = cv.threshold(blur_image, 0, 255, cv.THRESH_BINARY + cv.THRESH_TRIANGLE)

    def __align_image(self):
        color_img = cv.cvtColor(self.__grey_scale_image, cv.COLOR_GRAY2RGB)
        cv.circle(color_img, (self.__tracking_object.coordinates.x, self.__tracking_object.coordinates.y), 7, (0, 0,
                                                                                                               255), -1)
        cv.drawContours(color_img, self.__tracking_object.contours, 0, (0, 255, 0), 3)

        for corner in self.__tracking_object.corners.__dict__.values():
            cv.circle(color_img, (corner.x, corner.y), 5, (255, 0, 0), 3)

        # draw bounding box
        cv.drawContours(color_img, self.__tracking_object.bounding_box, 0, (0, 0, 255), 2)

        image = rotate_image_around_center_of_mass(color_img,
                                                   self.__tracking_object.coordinates.orientation_degrees,
                                                   self.__tracking_object.coordinates.x,
                                                   self.__tracking_object.coordinates.y)

        self.__tracking_object.set_aligned_image(image)

    def __detect_object(self):
        self.__preprocess_image()

        contours, hierarchy = cv.findContours(self.__preprocessed_image, cv.RETR_TREE, cv.CHAIN_APPROX_SIMPLE)

        # find the longest contour
        cnt = max(contours, key=cv.contourArea)

        moments = Moments(self.__preprocessed_image)

        self.__tracking_object = Object(moments, [cnt])
        self.__align_image()

    def get_object(self, image):
        self.__grey_scale_image = image
        self.__detect_object()
        return self.__tracking_object


def rotate_image_around_center_of_image(image, angle):
    center_x = image.shape[1] / 2
    center_y = image.shape[0] / 2
    rotation_matrix = cv.getRotationMatrix2D((center_x, center_y), angle, 1.0)
    result = cv.warpAffine(image, rotation_matrix, image.shape[1::-1], flags=cv.INTER_LINEAR)
    return result


def rotate_image_around_center_of_mass(image, angle, center_x, center_y):
    rotation_matrix = cv.getRotationMatrix2D((center_x, center_y), angle, 1.0)
    result = cv.warpAffine(image, rotation_matrix, image.shape[1::-1], flags=cv.INTER_LINEAR)
    return result
