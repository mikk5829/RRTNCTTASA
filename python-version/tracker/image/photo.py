import os
from typing import Union, Any

import cv2 as cv
from cv2 import Mat
from numpy import ndarray, dtype, generic


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


def get_image_from_file(file_path) -> Union[ndarray, ndarray[Any, dtype[generic]]]:
    """
    This function is used to get an image from a file
    :param file_path: the path to the file with the image
    :return: the image
    """
    # read image with open cv
    img = cv.imread(file_path, cv.IMREAD_GRAYSCALE)

    if img is None:
        raise ValueError(f"The file {os.path.abspath(file_path)} is not an image.")

    # return image
    return img


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


def get_raw_images_from_directory_generator(folder_path) -> (str, cv.Mat):
    images_paths = get_files_from_directory(folder_path)
    for path in images_paths:
        try:
            image = get_image_from_file(os.path.join(folder_path, path))
        except ValueError:
            continue

        yield path, image


def get_images_from_directory(folder_path) -> dict[str, cv.Mat]:
    """
    This function is used to get all the images from a folder and return them in a dict with the filename as key
    :param folder_path: the path to the folder with the images
    :return: dict with images
    """
    images = dict()
    file_counter = 1
    # find files in folder
    for filename in sorted(os.listdir(folder_path)):
        # read images with open cv
        img = cv.imread(os.path.join(folder_path, filename),
                        cv.IMREAD_GRAYSCALE)  # TODO might be better to skip this step and only read the images when needed
        filename_number = filename.split(".")[0]

        if img is not None:
            # make sure the filename_number is a number and increasing each iteration
            try:
                int(filename_number)
            except ValueError:
                raise ValueError(f"The file {filename} is not named correctly. It should be a number.")
            if int(filename_number) != file_counter:
                raise ValueError("The images in the folder are not named correctly. "
                                 "The images should be named in ascending order starting from 1. "
                                 f"The current image has number: {filename_number}, but should have number: {file_counter}")

            # append images to list
            images[filename] = img
            file_counter += 1

    # return dict with images
    return images


# A function that yields the images from a folder one by one
def get_images_from_directory_generator(folder_path) -> (str, cv.Mat):
    images = get_images_from_directory(folder_path)
    for key in images.keys():
        yield (key, images[key])
