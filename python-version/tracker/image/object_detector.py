import cv2 as cv

from models.moments import Moments
from models.object import Object


class ObjectDetector:
    tracking_object: Object = None
    grey_scale_image = None
    preprocessed_image = None
    aligned_image = None

    def __init__(self):
        # reset the state
        self.tracking_object = None
        self.grey_scale_image = None
        self.preprocessed_image = None

    def preprocess_image(self):
        # make a gaussian blur of the image to remove noise
        blur_image = cv.GaussianBlur(self.grey_scale_image, (5, 5), 0)

        # make a threshold of the blurred image to get a binary image
        ret, self.preprocessed_image = cv.threshold(blur_image, 30, 255, cv.THRESH_BINARY)

    def align_image(self):
        color_img = cv.cvtColor(self.grey_scale_image, cv.COLOR_GRAY2RGB)
        cv.circle(color_img, (self.tracking_object.coordinates.x, self.tracking_object.coordinates.y), 7, (0, 0,
                                                                                                           255), -1)
        cv.drawContours(color_img, self.tracking_object.contours, 0, (0, 255, 0), 3)

        image = rotate_image_around_center_of_mass(color_img,
                                                   self.tracking_object.coordinates.orientation_degrees,
                                                   self.tracking_object.coordinates.x,
                                                   self.tracking_object.coordinates.y)

        self.tracking_object.set_aligned_image(image)

    def detect_object(self):
        self.preprocess_image()

        contours, hierarchy = cv.findContours(self.preprocessed_image, cv.RETR_TREE, cv.CHAIN_APPROX_SIMPLE)

        # find the longest contour
        cnt = max(contours, key=cv.contourArea)

        moments = Moments(self.preprocessed_image)

        self.tracking_object = Object(moments, [cnt])
        self.align_image()

    def get_object(self, image):
        self.grey_scale_image = image
        self.detect_object()
        return self.tracking_object


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
