import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from scipy.signal import correlate, find_peaks
from scipy.signal import correlation_lags

from image.image_service import ImageService
from image.object_service import ObjectService
from image.pose_map_service import PoseMapService
from models.moments import Coordinates
from models.pose import Rotation, Translation
from service.service_interface import IService
import cv2 as cv
from scipy.spatial import distance
from fastdtw import fastdtw

from test import ResizeWithAspectRatio


def shift_using_peaks(x, y):
    # find the index of the highest value in x and y
    x_max_index = x.idxmax()
    y_max_index = y.idxmax()

    # find the index of the lowest value in x and y
    x_min_index = x.idxmin()
    y_min_index = y.idxmin()

    # find the difference between the two
    max_diff = x_max_index - y_max_index
    min_diff = x_min_index - y_min_index
    return (max_diff + min_diff) / 2


def shift_for_maximum_correlation(x, y):
    # round the index in the series to integers
    x = pd.Series(x.values, index=x.index.astype(int))
    y = pd.Series(y.values, index=y.index.astype(int))
    # drop duplicates index
    x = x[~x.index.duplicated(keep='first')]
    y = y[~y.index.duplicated(keep='first')]
    # # fill missing values with Nan from 0 - 360
    x = x.reindex(range(0, 360))
    y = y.reindex(range(0, 360))
    # fill with the nearest value
    x = x.interpolate(method="nearest")
    y = y.interpolate(method="nearest")

    x = pd.concat([x, x, x], ignore_index=True)
    y = pd.concat([y, y, y], ignore_index=True)

    # fill nan with 0 from 0 - 360
    # x = x.ffill().bfill()
    # y = y.ffill().bfill()
    x = x.fillna(0)
    y = y.fillna(0)
    # append the series to itself

    # get x values as list
    # x = x.values.tolist()
    # get y values as list
    # y = y.values.tolist()
    # konverter til 360 punkter
    # fyld manglende værdier med den nærmeste værdi

    # make x and y a 2d list with the index as the first element
    # Series to 2d array with index as first element
    x = np.array([x.index.values, x.values]).T
    y = np.array([y.index.values, y.values]).T

    d, path = fastdtw(y, x, dist=distance.euclidean)
    d %= len(x)
    d %= 360

    numeric_distances = [distance.braycurtis, distance.canberra, distance.chebyshev, distance.cityblock,
                         distance.correlation,
                         distance.cosine, distance.euclidean,
                         distance.jensenshannon, distance.mahalanobis, distance.minkowski, distance.seuclidean,
                         distance.sqeuclidean]
    for numeric_distance in numeric_distances:
        try:
            d_local, path = fastdtw(x, y, dist=numeric_distance)
            d_local %= len(x)
            d_local %= 360
            print(f"{numeric_distance.__name__}: {d_local}")
        except Exception as e:
            continue

    # x = pd.Series(x[1], index=x[0])
    # y = pd.Series(y[1], index=y[0])
    # correlation = correlate(x, y, mode="full")
    # lags = correlation_lags(x.size, y.size, mode="full")
    # lag = lags[np.argmax(correlation)]
    # print(f"Best lag: {lag}")
    # # find angle difference
    # angle_diff = x.iloc[0] - x.iloc[lag]
    # new_index = x.index + angle_diff

    # if lag < 0:
    #     y = y.iloc[abs(lag):].reset_index()
    # else:
    #     x = pd.Series(x.iloc[lag:].values)
    #     # x = x.iloc[lag:].reset_index()
    return d


def shift_own_method(x, y):
    x = pd.Series(x[1], index=x[0])
    y = pd.Series(y[1], index=y[0])
    # find peaks in x
    peaks_x, _ = find_peaks(x, prominence=1)  # find_peaks(x, prominence=1, width=3, distance=5, height=0.7 * max(
    # x.values))
    # find peaks in y
    peaks_y, _ = find_peaks(y, prominence=1)  # find_peaks(y, prominence=1, width=3, distance=5, height=0.7 * max(
    # y.values))
    # return the values in x and y at the peaks
    peaks_x = x.iloc[peaks_x]
    peaks_y = y.iloc[peaks_y]
    return peaks_x, peaks_y


class ContourService(IService):
    image_path = None
    model_name = None
    __image_service: ImageService = None
    __object_service: ObjectService = None
    __pose_map_service: PoseMapService = None

    def __init__(self, config, image_service, object_service, pose_map_service):
        super().__init__(config)
        self.__image_service = image_service
        self.__object_service = object_service
        self.__pose_map_service = pose_map_service

    def get_best_match(self, plot_best=False) -> (Rotation, cv.UMat):
        tracked_object = self.__object_service.get_object()
        contours = tracked_object.get_contour()

        # match the contours to the model
        pose_map = self.__pose_map_service.get_pose_map()
        best_score = 1  # initialize with the worst score
        found_pose = None
        found_model_contour = None

        for rotation, model_contour in pose_map.items():
            local_score = cv.matchShapes(contours, model_contour, 1, 0.0)

            if best_score > local_score:  # if the score is better than the previous best score
                best_score = local_score  # set the new best score
                found_pose = rotation  # set the new best translation
                found_model_contour = model_contour  # set the new best model contour

        # find roll when pitch and yaw are known
        # find COM from the model contour
        com_model_contour = cv.moments(contours)
        com_model_contour_x = int(com_model_contour["m10"] / com_model_contour["m00"])
        com_model_contour_y = int(com_model_contour["m01"] / com_model_contour["m00"])

        # find COM from the model contour
        com_found_model_contour = cv.moments(found_model_contour)
        com_found_model_contour_x = int(com_found_model_contour["m10"] / com_found_model_contour["m00"])
        com_found_model_contour_y = int(com_found_model_contour["m01"] / com_found_model_contour["m00"])
        #
        if plot_best:
            fig = plt.figure()
            ax = fig.add_subplot()
            # plot the best match
            plt.plot(contours[:, 0, 0], contours[:, 0, 1], "r")
            plt.plot(found_model_contour[:, 0, 0], found_model_contour[:, 0, 1], "b")
            plt.title(f"Contour matching", wrap=True)
            plt.suptitle(f"Score: {best_score:.2f}, {found_pose}", fontsize=10)
            plt.xlabel("x")
            plt.ylabel("y")
            plt.plot(com_model_contour_x, com_model_contour_y, "r*")
            plt.plot(com_found_model_contour_x, com_found_model_contour_y, "b*")
            plt.legend(["Model", "Found", "COM", "Found COM"])
            plt.show()

        # find the distance from the COM to all points in the model contour
        # calculate the distance between the COM and the points in the contours
        angle = np.arctan2(contours[:, 0, 1] - com_model_contour_y,
                           contours[:, 0, 0] - com_model_contour_x)
        dist = np.sqrt((com_model_contour_x - contours[:, 0, 0]) ** 2 + (
                com_model_contour_y - contours[:, 0, 1]) ** 2)

        # calculate the angle between the COM and the points in the contours
        angle_found = np.arctan2(found_model_contour[:, 0, 1] - com_found_model_contour_y,
                                 found_model_contour[:, 0, 0] - com_found_model_contour_x)
        dist_found = np.sqrt((com_found_model_contour_x - found_model_contour[:, 0, 0]) ** 2 + (
                com_found_model_contour_y - found_model_contour[:, 0, 1]) ** 2)

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

        return found_pose, found_model_contour
