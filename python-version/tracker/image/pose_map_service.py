import math
import os

import pickle

import numpy as np
import pandas as pd
import cv2 as cv
from matplotlib import pyplot as plt
from matplotlib import cm

from image.image_service import ImageService
from image.object_service import ObjectService
from models.pose import Pose, Rotation, Translation
from service.service_interface import IService


class PoseMapService(IService):
    """
    This class is used to create and read a pose map
    """
    path_to_model_images = None
    model_name = None
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
        image_generator = self.__image_service.get_raw_images_from_directory_generator()
        pose_map = dict()
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
                pose_map[pose] = tracked_object.get_contour()
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
        og, og_contour = list(pose_map.items())[-1]

        # conver phi and theta to x,y,z
        for rotation, model_contour in pose_map.items():
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

        # calculate and save the score for each rotation
        # data = []
        # for rotation1, model_contour1 in pose_map.items():
        #     # temp = []
        #     for rotation2, model_contour2 in pose_map.items():
        #         local_score = cv.matchShapes(model_contour1, model_contour2, 1, 0.0)
        #         # save all in DataFrame
        #         data.append(
        #             {
        #                 "pitch1": rotation1.pitch,
        #                 "yaw1": rotation1.yaw,
        #                 "pitch2": rotation2.pitch,
        #                 "yaw2": rotation2.yaw,
        #                 "score": local_score
        #             }
        #         )
        #     #     temp.append(local_score)
        #     # data[rotation1] = temp

        keys = list(pose_map.keys())

        # set index to keys
        df = pd.DataFrame(data)

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
        # c = c(df['score'])

        ax.view_init(og.pitch, og.yaw, 0)
        ax.scatter(x, y, z, c=c, cmap=colour)

        plt.show()

        # def random_point(r=1):
        #     ct = 2 * np.random.rand() - 1
        #     st = np.sqrt(1 - ct ** 2)
        #     phi = 2 * np.pi * np.random.rand()
        #     x = r * st * np.cos(phi)
        #     y = r * st * np.sin(phi)
        #     z = r * ct
        #     return np.array([x, y, z])
        #
        # def near(p, pntList, d0):
        #     cnt = 0
        #     for pj in pntList:
        #         dist = np.linalg.norm(p - pj)
        #         if dist < d0:
        #             cnt += 1 - dist / d0
        #     return cnt
        #
        # pointList = np.array([random_point(10.05) for i in range(65)])
        #
        # fig = plt.figure()
        # ax = fig.add_subplot(1, 1, 1, projection='3d')
        #
        # u = np.linspace(0, 2 * np.pi, 120)
        # v = np.linspace(0, np.pi, 60)
        #
        # # create the sphere surface
        # XX = 10 * np.outer(np.cos(u), np.sin(v))
        # YY = 10 * np.outer(np.sin(u), np.sin(v))
        # ZZ = 10 * np.outer(np.ones(np.size(u)), np.cos(v))
        #
        # WW = XX.copy()
        # for i in range(len(XX)):
        #     for j in range(len(XX[0])):
        #         x = XX[i, j]
        #         y = YY[i, j]
        #         z = ZZ[i, j]
        #         WW[i, j] = near(np.array([x, y, z]), pointList, 3)
        #     WW = WW / np.amax(WW)
        #     myheatmap = WW
        #
        #     # ~ ax.scatter( *zip( *pointList ), color='#dd00dd' )
        #     ax.plot_surface(XX, YY, ZZ, cstride=1, rstride=1, facecolors=cm.jet(myheatmap))
        #     plt.show()
        # make a pivot table from pitch and yaw
        # df = df.pivot(index="pitch", columns="yaw", values="score")
        # make plot showing the score and the distance
        # df.plot.scatter(x="x", y="y", z="z", c="score")

        # df.plot.scatter(x="dis", y="pitch", c="score", colormap='viridis')
        # make a heatmap
        # df = df.pivot(index="yaw1", columns="pitch1", values="score")
        # df = df.reset_index().pivot(index="pitch1", columns="pitch2", values="score")

        # plt.imshow(df)
        # plt.show()
        # save to csv
        df.to_csv('pose_map_data.csv')

        # save the pose map to a pickle file
        self.__pose_map = pose_map
        file = open(self.__pickle_name, "wb")
        pickle.dump(self.__pose_map, file)
        file.close()
        print("Pose map created successfully.")
        return True
