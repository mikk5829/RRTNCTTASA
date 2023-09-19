import math

import cv2 as cv


class Coordinates:
    """
    This class is used to store the coordinates of an object
    """
    x: int = None
    y: int = None
    orientation: float = None
    orientation_degrees: float = None

    def __init__(self, x: int, y: int, orientation: float = None):
        self.x = x
        self.y = y
        self.orientation = orientation
        self.orientation_degrees = orientation * 180 / math.pi

    def __str__(self):
        return f"x: {self.x}, y: {self.y}, orientation in radians: {self.orientation}, orientation in degrees:" \
               f" {self.orientation_degrees}"


class Moments:
    """
    This class is used to calculate the moments of an object
    """
    binary_image = None
    __coordinates: Coordinates = None

    def __init__(self, binary_image):
        self.binary_image = binary_image
        self.__set_coordinates()

    def __set_coordinates(self):
        center_of_mass = cv.moments(self.binary_image)
        x = int(center_of_mass["m10"] / center_of_mass["m00"])
        y = int(center_of_mass["m01"] / center_of_mass["m00"])
        orientation = math.atan2(center_of_mass["mu11"], center_of_mass["mu20"] - center_of_mass["mu02"]) / 2
        self.__coordinates = Coordinates(x, y, orientation)

    def get_coordinates(self):
        return self.__coordinates
