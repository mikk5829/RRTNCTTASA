import os
import pickle

from image.image_service import ImageService
from image.object_service import ObjectService
from models.object import Object
from models.pose import Rotation
from service.service_interface import IService


class PoseMapService(IService):
    """
    This class is used to create and read a pose map
    """
    path_to_model_images = None
    model_name = None
    verbose = False
    __pose_map = None
    __pickle_name = "pose_map.pickle"
    __object_service: ObjectService = None
    __image_service: ImageService = None

    def __init__(self, config, object_service, image_service):
        super().__init__(config)
        self.__object_service = object_service
        self.__image_service = image_service
        # create the folder for the model if it does not exist
        if not os.path.exists(self.model_name):
            os.mkdir(self.model_name)
        self.__pickle_name = self.model_name + "/" + self.__pickle_name

    def get_pose_map(self) -> dict[Rotation, Object]:
        if self.__pose_map is None:
            self.__pose_map = self.__pose_map_from_file()
            return self.__pose_map
        else:
            return self.__pose_map

    def __pose_map_from_file(self):
        # read the pose map from pickle file
        try:
            with open(self.__pickle_name, "rb") as f:
                pose_map = pickle.load(f)
            return pose_map
        except OSError:
            # if not available, prompt the user to create one using cli
            raise FileNotFoundError("No pose map found. Please create one using the cli.")

    def set_new_pose_map(self):
        """
        This function is used to create a new pose map from the images in the folder, and make a pickle file
        containing the pose map
        :return: True if successful
        """
        if self.path_to_model_images is None:
            raise ValueError("No path to model images provided. Please provide one using the cli.")
        image_generator = self.__image_service.get_raw_images_from_directory_generator()
        pose_map: dict[Rotation, Object] = dict()
        # create a new pose map from the images in the folder
        while image_generator:
            try:
                # create a new pose map from the images in the folder
                key, image = next(image_generator)
                tracked_object = self.__object_service.get_object(is_model=True)
                channel, theta, phi = key.split("_")
                phi = phi.split(".png")[0]
                rotation = Rotation(None, float(phi), float(theta))
                tracked_object.set_rotation(rotation)
                pose_map[rotation] = tracked_object  # .get_contour()
                # save the pose map to a pickle file
            except StopIteration:
                break
            except ValueError:
                print(f"Skipping image {key}")
                continue

        # sort the pose map by rotation
        pose_map = dict(sorted(pose_map.items()))

        # save the pose map to a pickle file
        self.__pose_map = pose_map
        print("Compressing pose map...")
        with open(self.__pickle_name, "wb") as f:
            pickle.dump(pose_map, f)
        print("Pose map created successfully.")
        return True
