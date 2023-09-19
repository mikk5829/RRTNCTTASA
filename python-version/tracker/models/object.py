from models.moments import Coordinates, Moments, SimpleCoordinates
import cv2 as cv


class Corners:
    top_left: SimpleCoordinates = None
    top_right: SimpleCoordinates = None
    bottom_left: SimpleCoordinates = None
    bottom_right: SimpleCoordinates = None

    def __init__(self, top_left: SimpleCoordinates, top_right: SimpleCoordinates, bottom_left: SimpleCoordinates,
                 bottom_right: SimpleCoordinates):
        self.top_left = top_left
        self.top_right = top_right
        self.bottom_left = bottom_left
        self.bottom_right = bottom_right


class Object:
    coordinates: Coordinates = None
    moments: Moments = None
    contours = None
    aligned_image = None
    corners: Corners = None
    bounding_box = None

    def __init__(self, moments: Moments, contours):
        self.moments = moments
        self.coordinates = moments.get_coordinates()
        self.contours = contours
        self.__set_corners()

    def __set_corners(self):
        ct = self.contours[0].squeeze()
        # find the top left corner
        bottom_left = SimpleCoordinates(ct[ct[:, 0].argmin()])
        top_right = SimpleCoordinates(ct[ct[:, 0].argmax()])
        top_left = SimpleCoordinates(ct[ct[:, 1].argmin()])
        bottom_right = SimpleCoordinates(ct[ct[:, 1].argmax()])

        self.corners = Corners(top_left, top_right, bottom_left, bottom_right)

        # find the bounding box
        min_area_rect = cv.minAreaRect(self.contours[0])
        bounding_box = cv.boxPoints(min_area_rect)
        bounding_box = bounding_box.astype('int')

        self.bounding_box = [bounding_box]

    def set_aligned_image(self, image):
        self.aligned_image = image

    def __str__(self):
        return f"Coordinates: {self.coordinates}"
