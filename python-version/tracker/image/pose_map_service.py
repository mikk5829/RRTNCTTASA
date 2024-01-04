import math
import os
import pickle
import cv2 as cv
import pandas as pd
from tqdm import tqdm

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

    def get_pose_map(self) -> dict[int, Object]:
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
        pose_map = dict()
        # create a new pose map from the images in the folder
        while image_generator:
            try:
                # create a new pose map from the images in the folder
                key, image = next(image_generator)
                tracked_object = self.__object_service.get_object(is_model=True)
                channel, theta, phi, index, furthest_index = key.split("_")
                furthest_index = furthest_index.split(".png")[0]
                furthest_index = int(float(furthest_index))
                index = int(float(index))
                theta = float(theta)
                phi = float(phi)
                # calculate the cartesian coordinates of the object
                # x = math.sin(theta) * math.cos(phi)
                # y = math.sin(theta) * math.sin(phi)
                # z = math.cos(theta)
                # # our coordinate system is rotated, so we need to rotate it back
                # x, y, z = x, -z, y
                # # find the roll of the object
                # roll = math.atan2(y, x)
                # roll = math.degrees(roll)

                rotation = Rotation(None, theta, phi, channel)
                tracked_object.set_rotation(rotation)
                tracked_object.set_furthest_index(str(furthest_index) + channel)
                pose_map[str(index) + channel] = tracked_object  # .get_contour()
                # save the pose map to a pickle file
            except StopIteration:
                break
            except ValueError as e:
                print(e)
                print(f"Skipping image {key}")
                continue

        # match every contour with every contour and store in pandas
        # data = []
        # for key, tracked_object in tqdm(pose_map.items()):
        #     for key2, tracked_object2 in pose_map.items():
        #         local_score = cv.matchShapes(tracked_object.get_contour(), tracked_object2.get_contour(), 1, 0.0)
        #         data.append([key, key2, local_score])
        #
        # df = pd.DataFrame(data, columns=["key", "key2", "score"])
        # df.to_csv("scores.csv")
        #
        # # make pivot table
        # df_pivot = df.pivot(index="key", columns="key2", values="score")
        # # plot the pivot table
        # df_pivot.plot()

        # save the pose map to a pickle file
        self.__pose_map = pose_map
        print("Compressing pose map...")
        with open(self.__pickle_name, "wb") as f:
            pickle.dump(pose_map, f)
        print("Pose map created successfully.")
        return True
