import cv2 as cv

from image.image_service import ImageService
from models.object import Object
from service.service_interface import IService


# Image to Object
class ObjectService(IService):
    __tracking_object: Object = None
    __image_service: ImageService = None
    verbose = False

    def __init__(self, config, image_service):
        super().__init__(config)
        self.__image_service = image_service

    def get_object(self, is_model=False) -> Object:
        # if self.__tracking_object is None or is_model:
        self.__tracking_object = Object(self.__image_service.get_image(), True, verbose=self.verbose,
                                        is_model=is_model, file_name=self.__image_service.image_path)
        return self.__tracking_object

    def set_object(self, tracked_object):
        self.__tracking_object = tracked_object


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
