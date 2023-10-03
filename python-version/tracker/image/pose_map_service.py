import os

import pickle

from image.image_service import ImageService
from image.object_service import ObjectService
from models.pose import Translation
from service.service_interface import IService


class PoseMapService(IService):
    """
    This class is used to create and read a pose map
    """
    path_to_model_images = None
    model_name = None
    __pose_map = None
    __pickle_name = "pose_map.pickle"
    object_service: ObjectService = None
    image_service: ImageService = None

    def __init__(self, config, object_service, image_service):
        super().__init__(config)
        self.object_service = object_service
        self.image_service = image_service
        # create the folder for the model if it does not exist
        if not os.path.exists(self.model_name):
            os.mkdir(self.model_name)
        self.__pickle_name = self.model_name + "/" + self.__pickle_name
        if self.path_to_model_images is not None:
            self.set_new_pose_map()

    def get_pose_map(self):
        if self.__pose_map is None:
            self.__pose_map = self.__pose_map_from_file()
            return self.__pose_map
        else:
            return self.__pose_map

    def __pose_map_from_file(self):
        # read the pose map from pickle file
        try:
            file = open(self.__pickle_name, "rb")
            pose_map = pickle.load(file)
            file.close()
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
        image_generator = self.image_service.get_raw_images_from_directory_generator()
        pose_map = dict()
        # create a new pose map from the images in the folder
        while image_generator:
            try:
                # create a new pose map from the images in the folder
                key, image = next(image_generator)
                self.object_service.get_object()
                tracked_object = self.object_service.get_object(image)
                # change -0.000_-1.000_0.000.png to x, y, z from key
                x, y, z = key.split("_")
                z = z.split(".png")[0]
                translation = Translation(x, y, z)
                pose_map[translation] = tracked_object.contours
                # save the pose map to a pickle file
            except StopIteration:
                break
            except ValueError:
                print(f"Skipping image {key}")
                continue
        # save the pose map to a pickle file
        self.__pose_map = pose_map
        file = open(self.__pickle_name, "wb")
        pickle.dump(self.__pose_map, file)
        file.close()
        return True
