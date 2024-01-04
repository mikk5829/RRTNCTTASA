from matplotlib import pyplot as plt
from tqdm import tqdm

from image.image_service import ImageService, resize_with_aspect_ratio
import cv2 as cv

from models.object import Object
from service.service_interface import IService


class Tracker(IService):
    """
    This class is used to track an object in an image
    """
    __estimations = None  # List of old estimations TODO read from file
    image_path: str = None  # Path to the image
    __object_service = None  # Service to get objects
    __contour_service = None  # Service to get contours
    __image_service: ImageService = None

    def __init__(self, config, object_service, contour_service, image_service):
        super().__init__(config)
        self.__object_service = object_service
        self.__contour_service = contour_service
        self.__image_service = image_service

    def estimate_poses(self):
        image_generator = self.__image_service.get_raw_images_from_directory_generator()
        while image_generator:
            try:
                # create a new pose map from the images in the folder
                key, image = next(image_generator)
                self.__contour_service.get_best_match()
            except StopIteration:
                break
            except ValueError as e:
                print(e)
                print(f"Skipping image {key}")
                continue

    def estimate_pose(self):
        """
        This function is used to estimate the pose of an object
        :return: The pose of the object
        """
        # if self.image_path is None:
        #     exit("No image path provided. Please provide one using the cli.")
        # Todo when estimations are present, use them to print the last pose
        # Find the object in the image
        tracked_object = self.__object_service.get_object()

        # self.contour_service.simplify_contours()
        self.__contour_service.get_best_match()

        # model_contour_length = len(model)
        #
        # x = 0
        # y = 0
        # x_values = []
        # y_values = []
        #
        # for l in model:
        #     x = x + l[0][0]
        #     y = y + l[0][1]
        #     x_values.append(l[0][0])
        #     y_values.append(l[0][1])
        #
        # x /= model_contour_length
        # y /= model_contour_length
        #
        # print(x, y)

        # find the length from x,y to all points in the contour

        # plt.plot(x_values, y_values, "b")
        # plt.show()

        # find

        # Compare the object to the 3D (mesh) model if available also use old information to predict the pose
        # Estimate, save and return the pose
        # img = tracked_object.get_aligned_image()
        # cv.drawContours(img, model, -1, (255, 0, 0), 3)
        # cv.imshow("image", resize_with_aspect_ratio(tracked_object.get_raw_image(), width=800))
        # cv.imshow("tracker", resize_with_aspect_ratio(img, width=800))
        # cv.waitKey()
        return None
