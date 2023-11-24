import cv2 as cv
import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from scipy.signal import find_peaks

from image.image_service import ImageService
from image.object_service import ObjectService
from image.pose_map_service import PoseMapService
from models.object import Object
from models.pose import Pose, Rotation
from service.service_interface import IService


def shift_using_peaks(x, y):
    # find the index of the highest value in x and y
    x_max_index = x.idxmax()
    y_max_index = y.idxmax()

    # find the index of the lowest value in x and y
    x_min_index = x.idxmin()
    y_min_index = y.idxmin()

    # find the difference between the two
    max_diff = x_max_index - y_max_index
    return max_diff


# def shift_own_method(x, y):
#     x = pd.Series(x[1], index=x[0])
#     y = pd.Series(y[1], index=y[0])
#     # find peaks in x
#     peaks_x, _ = find_peaks(x, prominence=1)  # find_peaks(x, prominence=1, width=3, distance=5, height=0.7 * max(
#     # x.values))
#     # find peaks in y
#     peaks_y, _ = find_peaks(y, prominence=1)  # find_peaks(y, prominence=1, width=3, distance=5, height=0.7 * max(
#     # y.values))
#     # return the values in x and y at the peaks
#     peaks_x = x.iloc[peaks_x]
#     peaks_y = y.iloc[peaks_y]
#     return peaks_x, peaks_y


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

    def get_best_match(self) -> (Rotation, cv.UMat):
        tracked_object = self.__object_service.get_object()
        contour = tracked_object.get_contour()

        # match the contours to the model
        pose_map = self.__pose_map_service.get_pose_map()
        best_score = 1  # initialize with the worst score
        found_pose: Pose
        found_object: Object = None

        for pose, tracking_object in pose_map.items():
            local_score = cv.matchShapes(contour, tracking_object.get_contour(), 1, 0.0)

            if best_score > local_score:  # if the score is better than the previous best score
                best_score = local_score  # set the new best score
                found_pose = pose  # set the new best translation
                found_object = tracking_object  # set the new best model contour

        # find COM from the model contour
        found_model_contour = found_object.get_contour()

        if self.verbose:
            fig = plt.figure()
            ax = fig.add_subplot()
            # plot the best match
            plt.plot(contour[:, 0, 0], contour[:, 0, 1], "r")
            plt.plot(found_model_contour[:, 0, 0], found_model_contour[:, 0, 1], "b")
            plt.title(f"Contour matching", wrap=True)
            plt.suptitle(f"Score: {best_score:.2f}, {found_pose}", fontsize=10)
            plt.xlabel("x")
            plt.ylabel("y")
            plt.plot(tracked_object.coordinates.x, tracked_object.coordinates.y, "r*")
            plt.plot(found_object.coordinates.x, found_object.coordinates.y, "b*")
            plt.legend(["Model", "Found", "COM", "Found COM"])
            plt.show()

        # find the distance from the COM to all points in the model contour
        # calculate the distance between the COM and the points in the contours
        angle = np.arctan2(contour[:, 0, 1] - tracked_object.coordinates.y,
                           contour[:, 0, 0] - tracked_object.coordinates.x)
        dist = np.sqrt((tracked_object.coordinates.x - contour[:, 0, 0]) ** 2 + (
                tracked_object.coordinates.y - contour[:, 0, 1]) ** 2)

        # calculate the angle between the COM and the points in the contours
        angle_found = np.arctan2(found_model_contour[:, 0, 1] - found_object.coordinates.y,
                                 found_model_contour[:, 0, 0] - found_object.coordinates.x)
        dist_found = np.sqrt((found_object.coordinates.x - found_model_contour[:, 0, 0]) ** 2 + (
                found_object.coordinates.y - found_model_contour[:, 0, 1]) ** 2)

        angle_found = np.rad2deg(angle_found) + 180
        angle = np.rad2deg(angle) + 180

        # sort angles from smallest to largest and sort the distances accordingly
        angle, dist = zip(*sorted(zip(angle, dist)))
        angle_found, dist_found = zip(*sorted(zip(angle_found, dist_found)))

        # Spline interpolation of dist and angle, such that the length is that same as angle_found
        dist = np.interp(angle_found, angle, dist)
        angle = np.interp(angle_found, angle, angle)

        # convert to series
        pd_angle = pd.Series(dist, index=angle)
        pd_angle_found = pd.Series(dist_found, index=angle_found)

        # plot the distance and angle for
        plt.plot(angle, dist, "r")
        plt.plot(angle_found, dist_found, "b")
        # Wrap title in matplotlib plot
        plt.title(f"2D shape matching", wrap=True)
        plt.xlabel("Angle in degrees")
        plt.ylabel("Distance")
        plt.legend(["Model", "Found"])
        plt.show()

        # find the best match between the two
        # shift the found contour to the best match
        # best_match_dist, best_match_dist_found = shift_for_maximum_correlation(pd_angle, pd_angle_found)
        angle_diff = shift_using_peaks(pd_angle, pd_angle_found)

        print(f"roll is: {angle_diff:.2f}")

        # minus the angle_diff from angle and make sure the angle is between 0 and 360
        angle = (angle - angle_diff) % 360
        angle, dist = zip(*sorted(zip(angle, dist)))
        plt.plot(angle, dist, "r")
        plt.plot(angle_found, dist_found, "b")
        # Wrap title in matplotlib plot
        plt.title(f"2D shape matching", wrap=True)
        plt.xlabel("Angle in degrees")
        plt.ylabel("Distance")
        plt.legend(["Model", "Found"])
        plt.show()

        # set rotation TODO
        return found_pose, found_model_contour
