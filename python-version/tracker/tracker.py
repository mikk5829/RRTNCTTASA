from image.pose_map_service import PoseMapService
from image.object_detector import ObjectDetector
from image.image_service import get_raw_images_from_directory_generator, resize_with_aspect_ratio, get_image_from_file
import cv2 as cv

from models.pose import Pose


class Tracker:
    image = None  # Image of the object
    key: str = None  # Key of the image
    pose = Pose()  # Pose of the object in the image
    estimations = None  # List of old estimations
    pose_map = None  # Map of poses and images
    image_path = None  # Path to the image
    image_service = None  # Service to get images

    def __init__(self, pose_map_importer: PoseMapService, image_path, image_service):
        self.pose_map = pose_map_importer.get_pose_map()
        self.image_path = image_path
        self.image_service = image_service

    def estimate_pose_from_folder(self):
        image_generator = get_raw_images_from_directory_generator(self.image_path)
        while image_generator:
            try:
                self.key, self.image = next(image_generator)
                self.estimate_pose()
            except StopIteration:
                break
        return

    def estimate_pose(self):
        if self.image_path is None:
            exit("No image path provided. Please provide one using the cli.")
            # Todo when estimations are present, use them to print the last pose
        # Find the object in the image
        self.image = self.image_service.get_image()
        object_detector = ObjectDetector()
        object = object_detector.get_object(self.image)
        print(object)
        cv.imshow("image", resize_with_aspect_ratio(self.image, width=800))
        cv.imshow("tracker", resize_with_aspect_ratio(object.aligned_image, width=800))
        cv.waitKey()
        # Either separate object from background or use line detection and choose the best fit
        # Compare the object to the 3D (mesh) model if available also use old information to predict the pose
        # Estimate, save and return the pose
        return self.pose
