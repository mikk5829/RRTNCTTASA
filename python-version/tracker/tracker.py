from image.pose_map_service import PoseMapService
from image.object_service import ObjectService
from image.image_service import resize_with_aspect_ratio
import cv2 as cv

from models.pose import Pose
from service.service_interface import IService


class Tracker(IService):
    """
    This class is used to track an object in an image
    """
    image = None  # Image of the object
    key: str = None  # Key of the image
    estimations = None  # List of old estimations TODO read from file
    image_path = None  # Path to the image
    object_service = None  # Service to get objects
    contour_service = None  # Service to get contours

    def __init__(self, config, object_service, contour_service):
        super().__init__(config)
        self.object_service = object_service
        self.contour_service = contour_service

    def estimate_pose(self):
        """
        This function is used to estimate the pose of an object
        :return: The pose of the object
        """
        if self.image_path is None:
            exit("No image path provided. Please provide one using the cli.")
            # Todo when estimations are present, use them to print the last pose
        # Find the object in the image
        tracked_object = self.object_service.get_object()

        # Compare the object to the 3D (mesh) model if available also use old information to predict the pose
        # Estimate, save and return the pose

        cv.imshow("image", resize_with_aspect_ratio(tracked_object.raw_image, width=800))
        cv.imshow("tracker", resize_with_aspect_ratio(tracked_object.aligned_image, width=800))
        cv.waitKey()
        return tracked_object.pose
