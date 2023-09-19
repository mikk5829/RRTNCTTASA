from cli.parser import get_image_path
from image.object_detector import ObjectDetector
from image.photo import get_images_from_directory_generator, resize_with_aspect_ratio, get_image_from_file
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
        self.image = get_image_from_file(get_image_path())
        object_detector = ObjectDetector()
        object = object_detector.get_object(self.image)
        print(object)
        cv.imshow("Source", resize_with_aspect_ratio(object.aligned_image, width=800))
        cv.waitKey()
        # Either separate object from background or use line detection and choose the best fit
        # Compare the object to the 3D (mesh) model if available also use old information to predict the pose
        # Estimate, save and return the pose
        return self.pose
