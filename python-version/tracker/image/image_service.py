import os
from typing import Union, Any

import cv2 as cv
from numpy import ndarray, dtype, generic

from service.service_interface import IService


class ImageService(IService):
    image_path = None
    model_name = None

    def __init__(self, config):
        super().__init__(config)

    def get_image(self) -> Union[ndarray, ndarray[Any, dtype[generic]]]:
        """
        This function is used to get an image from a file
        :path_override: The path to the image
        """
        # read image with open cv
        img = cv.imread(self.image_path, cv.IMREAD_GRAYSCALE)

        if img is None:
            raise ValueError(f"The file {os.path.abspath(self.image_path)} is not an image.")

        # return image
        return img

    def get_raw_images_from_directory_generator(self):
        images_paths = get_files_from_directory(self.path_to_model_images)
        for path in images_paths:
            try:
                self.image_path = (self.path_to_model_images + "/" + path)
                image = self.get_image()
            except ValueError:
                continue

            yield path, image


def resize_with_aspect_ratio(image, width=None, height=None, inter=cv.INTER_AREA):
    """
    This function is used to resize an image while keeping the aspect ratio
    :param image: Image to resize
    :param width: The width of the resized image
    :param height: The height of the resized image
    :param inter: The interpolation method
    :return: The resized image
    """
    dim = None
    (h, w) = image.shape[:2]

    if width is None and height is None:
        return image
    if width is None:
        r = height / float(h)
        dim = (int(w * r), height)
    else:
        r = width / float(w)
        dim = (width, int(h * r))

    return cv.resize(image, dim, interpolation=inter)


def get_files_from_directory(folder_path) -> list[str]:
    """
    This function is used to get all the files from a folder
    :param folder_path: the path to the folder with the files
    :return: list with files
    """
    files = list()
    # find files in folder
    for filename in sorted(os.listdir(folder_path)):
        # append files to list
        files.append(filename)

    # return list with files
    return files
