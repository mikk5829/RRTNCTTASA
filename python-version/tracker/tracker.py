from cli.parser import get_image_path
from image.photo import get_images_from_directory_generator
import cv2 as cv

from models.pose import Pose


class Tracker:
    image: cv.Mat = None  # Image of the object
    key: str = None  # Key of the image
    pose = Pose()  # Pose of the object in the image
    estimations = None  # List of old estimations

    def estimate_pose_from_folder(self):
        image_generator = get_images_from_directory_generator(get_image_path())
        while image_generator:
            try:
                self.key, self.image = next(image_generator)
                self.estimate_pose()
            except StopIteration:
                break
        return

    def estimate_pose(self):
        # Find the object in the image
        # Either separate object from background or use line detection and choose the best fit
        # Compare the object to the 3D (mesh) model if available also use old information to predict the pose
        # Estimate, save and return the pose
        return self.pose
