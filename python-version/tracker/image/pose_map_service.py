import lzma
import math
import gzip
import os

import pickle

import numpy as np
import pandas as pd
import cv2 as cv
from matplotlib import pyplot as plt
from matplotlib import cm

from image.image_service import ImageService
from image.object_service import ObjectService
from models.object import Object
from models.pose import Pose, Rotation, Translation
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

    def get_pose_map(self) -> dict[Pose, Object]:
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
        pose_map: dict[Pose, Object] = dict()
        # create a new pose map from the images in the folder
        while image_generator:
            try:
                # create a new pose map from the images in the folder
                key, image = next(image_generator)
                tracked_object = self.__object_service.get_object()
                theta, phi, x, y, z = key.split("_")
                z = z.split(".png")[0]
                rotation = Rotation(None, float(phi), float(theta))
                translation = Translation(float(x), float(y), float(z))
                pose = Pose(translation, rotation)
                pose_map[pose] = tracked_object  # .get_contour()
                # save the pose map to a pickle file
            except StopIteration:
                break
            except ValueError:
                print(f"Skipping image {key}")
                continue

        # sort the pose map by rotation
        pose_map = dict(sorted(pose_map.items()))

        # find the last item in the dict pose_map

        # make a df that contains the score for each rotation to the base rotation "rot" rows are pitch, columns are yaw
        data = []
        og_rotation, og_model = list(pose_map.items())[-1]

        # conver phi and theta to x,y,z
        for rotation, model_object in pose_map.items():
            model_contour = model_object.get_contour()
            og_contour = og_model.get_contour()
            # convert phi and theta to x,y,z
            x = rotation.x
            y = rotation.y
            z = rotation.z
            local_score = cv.matchShapes(og_contour, model_contour, 1, 0.0)
            # save all in DataFrame
            data.append(
                {
                    "x": x,
                    "y": y,
                    "z": z,
                    "score": local_score
                }
            )

        keys = list(pose_map.keys())

        # set index to keys
        df = pd.DataFrame(data)

        if self.verbose:
            fig = plt.figure()

            # syntax for 3-D projection
            ax = plt.axes(projection='3d')

            # defining axes
            z = df['z']
            x = df['x']
            y = df['y']
            c = df['score']
            # set a colour map for the scatter plot low score is green, high score is red
            colour = plt.cm.get_cmap('RdYlGn')

            # ax.view_init(og_rotation.pitch, og_rotation.yaw, 0)
            ax.scatter(x, y, z, c=c, cmap=colour)

            plt.show()

        df.to_csv('pose_map_data.csv')

        # save the pose map to a pickle file
        self.__pose_map = pose_map
        print("Compressing pose map...")
        with open(self.__pickle_name, "wb") as f:
            pickle.dump(pose_map, f)
        print("Pose map created successfully.")
        return True
