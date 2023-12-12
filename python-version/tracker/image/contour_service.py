import cv2 as cv
import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
import math

from image.image_service import ImageService
from image.object_service import ObjectService
from image.pose_map_service import PoseMapService
from models.object import Object
from models.pose import Pose, Rotation, Translation
from service.service_interface import IService


def custom_uniform_interpolation(angle, dist):
    # Custom uniform interpolation to make X uniformly distributed between 0 and 360 degrees with spacing of 1 degree.

    # Ensure that X is within [0, 360] and sorted in ascending order.
    angle = np.mod(angle, 360)
    sorted_idx = np.argsort(angle)
    angle = angle[sorted_idx]
    dist = dist[sorted_idx]

    # Prepare the data for interpolation.
    a = (dist[sorted_idx[0]] - dist[sorted_idx[-1]]) / abs(angle[sorted_idx[-1]] - angle[sorted_idx[0]] - 360)

    XY_end = [359, dist[sorted_idx[-1]] + a * (359 - angle[sorted_idx[-1]])]
    XY_1 = [0, dist[sorted_idx[0]] + a * (0 - angle[sorted_idx[0]])]

    angle = np.concatenate(([XY_1[0]], angle, [XY_end[0]]))
    dist = np.concatenate(([XY_1[1]], dist, [XY_end[1]]))

    # Initialize interpolated signal arrays.
    interpolated_X = np.arange(360)
    interpolated_Y = np.zeros(360)

    # Perform custom linear interpolation for each degree.
    for i, current_degree in enumerate(interpolated_X):
        # Find the closest points in X for interpolation.
        # print(np.size(X))
        try:
            lower_idx = np.where(angle <= current_degree)[0][-1]
        except:
            lower_idx = None
        try:
            upper_idx = np.where(angle > current_degree)[0][0]
        except:
            upper_idx = None

        if lower_idx is None or upper_idx is None:
            # If outside the original range, use the nearest endpoint.
            if current_degree <= angle[0]:
                lower_idx = 0
                upper_idx = 1
            else:
                lower_idx = len(angle) - 2
                upper_idx = len(angle) - 1

        # Interpolation weights.
        alpha = (current_degree - angle[lower_idx]) / (angle[upper_idx] - angle[lower_idx])

        # Custom linear interpolation.
        interpolated_Y[i] = (1 - alpha) * dist[lower_idx] + alpha * dist[upper_idx]

    return interpolated_Y


def XYZ(width, height, CoM, fs):
    m = 1
    mm = 1e-3
    um = 1e-6

    # An Effective Focal Length of:
    EFL_far = 20 * mm

    # and a pixel size on the CCD of:
    CCDy = 8.6 * um

    # What is the distance from the camera to the model in Blender?
    zm = 10 * m
    f = 20 * mm

    # The bigger the scale factor, the closer it is to us
    z = zm / fs

    cx = math.floor(width / 2)
    cy = math.floor(height / 2)
    px = ((CoM[0] - cx) * CCDy * z) / f
    py = ((CoM[1] - cy) * CCDy * z) / f

    return Translation([px, py, z])


def calculate_roll(found_object: Object, tracked_object: Object):
    curves = []
    for local_object in [found_object, tracked_object]:
        object_contour = local_object.get_contour()
        object_x = local_object.get_moments().get_coordinates().x
        object_y = local_object.get_moments().get_coordinates().y
        # Calculate the distance between every point and the CoM
        object_dist = np.sqrt((object_contour[:, :, 0] - object_x) ** 2 + (object_contour[:, :, 1] - object_y) ** 2)
        # normalize the distances
        object_dist -= np.min(object_dist)
        object_dist /= np.max(object_dist)
        # Calculate the angle between every point and the CoM
        object_angle = np.degrees(np.arctan2(object_contour[:, :, 1] - object_y,
                                             object_contour[:, :, 0] - object_x))
        # convert angles to degrees
        object_angle = np.mod(object_angle, 360)

        # sort angles from 0 to 360 and also the corresponding distances
        object_angle, object_dist = zip(*sorted(zip(object_angle, object_dist)))

        object_dist = np.asarray(object_dist).squeeze()
        object_angle = np.asarray(object_angle).squeeze()

        curves.append(custom_uniform_interpolation(object_angle, object_dist))

    # unpack the curves
    parametric_curve = curves[0]
    parametric_curve2 = curves[1]

    mult_sum = np.zeros(parametric_curve.shape[0])
    for i in range(len(mult_sum)):
        mult_sum[i] = np.dot(parametric_curve, np.roll(parametric_curve2, -i))

    # Find the index of the maximum value.
    roll = np.argmax(mult_sum)
    max_val = np.max(mult_sum)

    rotation = found_object.get_rotation()

    rotation.set_roll(roll)

    return rotation, max_val


class ContourService(IService):
    image_path = None
    model_name = None
    verbose = None
    __image_service: ImageService = None
    __object_service: ObjectService = None
    __pose_map_service: PoseMapService = None

    def __init__(self, config, image_service, object_service, pose_map_service):
        super().__init__(config)
        self.__image_service = image_service
        self.__object_service = object_service
        self.__pose_map_service = pose_map_service

    def get_best_match(self):
        tracked_object = self.__object_service.get_object()
        tracked_object_contour = tracked_object.get_contour()
        tracked_object_com = tracked_object.get_moments().get_coordinates().x, tracked_object.get_moments().get_coordinates().y
        # find de n bedste matches med contour
        # ud af dem, så lav krydskorrelation og find den bedste
        # returner det pose

        # match the contours to the model
        pose_map = self.__pose_map_service.get_pose_map()
        best_score = 1  # initialize with the worst score
        found_pose: Pose
        # list of objects
        found_objects: list[Object] = []

        for pose, tracking_object in pose_map.items():
            local_score = cv.matchShapes(tracked_object_contour, tracking_object.get_contour(), 1, 0.0)

            if local_score < 0.05:
                found_objects.append(tracking_object)

        found_rotation_list = []
        for found_object in found_objects:
            # plot the found_object, tracked_object
            plt.plot(tracked_object_contour[:, 0, 0], tracked_object_contour[:, 0, 1], "r")
            plt.plot(found_object.get_contour()[:, 0, 0], found_object.get_contour()[:, 0, 1], "b")
            plt.title(f"Contour matching", wrap=True)
            plt.suptitle(found_object.get_rotation(), wrap=True)
            plt.xlabel("x")
            plt.ylabel("y")
            plt.plot(tracked_object.coordinates.x, tracked_object.coordinates.y, "r*")
            plt.plot(found_object.coordinates.x, found_object.coordinates.y, "b*")
            plt.legend(["Image", "3D Model", "Image COM", "3D Model COM"])
            plt.show()
            found_rotation_list.append(calculate_roll(found_object, tracked_object))

        found_rotation_list = dict(found_rotation_list)

        # find the key with the highest value
        found_rotation = max(found_rotation_list, key=found_rotation_list.get)

        tracked_object.set_rotation(found_rotation)

        print(tracked_object.get_rotation())

        return 0
