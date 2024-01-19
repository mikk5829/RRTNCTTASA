import itertools
import time
import timeit

import matplotlib.pyplot as plt
import numpy as np
import math
import cv2 as cv
import pandas as pd
from scipy.optimize import minimize
from scipy.spatial import KDTree
import scipy.io
from tqdm import tqdm

from models.pose import Rotation, Translation

# 3D model points.
# three_d_points = np.array([
#     [0.35, -0.18, -0.01],  # Top right solar panel
#     [0.13, -0.18, -0.02],  # Top left solar panel
#     [0.35, 0.12, -0.01],  # Bottom right solar panel
#     [0.13, 0.12, -0.02],  # Bottom left solar panel
#     [-0.15, 0.09, -0.04],  # Bottom left satellite
#     [-0.15, -0.08, -0.04],  # Top left satellite
# ])

labels = ["TR", "TL", "BR", "BL", "BLS", "TLS"]


# im = cv.imread("test_images/dynamic_unknowndeg_0to360_5degstep/A240105_15341532.png")
# im = cv.imread("test_images/tesx_x.png")
# size = im.shape
# com = (364, 209)


def plot_3d(points):
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    # change the view such that y is down, x is right and z is in the screen
    ax.view_init(azim=-90, elev=-90)
    ax.scatter(points[:, 0], points[:, 1], points[:, 2])
    # for i, txt in enumerate(labels):
    #     ax.text(points[i][0], points[i][1], points[i][2], txt)
    # axis should have the same range -1 to 1
    limits = 3.5
    ax.set_xlim3d(-limits, limits)
    ax.set_ylim3d(-limits, limits)
    ax.set_zlim3d(-limits, limits)
    ax.set_xlabel('X Label')
    ax.set_ylabel('Y Label')
    ax.set_zlabel('Z Label')
    plt.show()


def plot_2d(points, extra_points=None, title="", legend=None, labels=None):
    # plot im moved such that com is in 0,0 extent=[horizontal_min,horizontal_max,vertical_min,vertical_max].
    plt.imshow(im, extent=[-size[1] // 2, size[1] // 2, size[0] // 2, -size[0] // 2])
    # plt.imshow(im)
    plt.title(title)
    # scatter plot points
    plt.scatter(points[:, 0], points[:, 1], c='r')
    # for point in points:
    #     plt.plot(point[0], point[1], 'ro')
    if extra_points is not None:
        plt.scatter(extra_points[:, 0], extra_points[:, 1], c='y')
        # for point in extra_points:
        #     plt.plot(point[0], point[1], 'bo')

    if labels is not None:
        for i, txt in enumerate(labels):
            plt.text(points[i][0], points[i][1], f"{txt:.2f}", c='r')
    if legend:
        plt.legend(legend)
    # plt.savefig(f"test_images/dynamic_unknowndeg_0to360_5degstep/minimize/{title}.png")
    plt.show()


# points in meters around the center of mass
three_d_points = np.array([
    [-0.08, -2.30333333, 1.19666667],  # Top right solar panel
    [-0.15666667, -0.84333333, 1.19666667],  # Top left solar panel
    [-0.08, -2.30333333, -0.78333333],  # Bottom right solar panel
    [-0.15666667, -0.84333333, -0.78333333],  # Bottom left solar panel
    [-0.24333333, 1.03, -0.65333333],  # Bottom left satellite
    [-0.24666667, 1.01333333, 0.69333333],  # Top left satellite
])

data = []
points = []
all_distances = []
iteration = 0


def match_three_d(two_d_points, weights, initial_guess):
    roll, pitch, yaw, x, y, z = initial_guess
    deg = 10
    trans_z = 1
    trans = 0.1
    bounds = (
        (roll - deg, roll + deg), (pitch - deg, pitch + deg), (yaw - deg, yaw + deg), (x - trans, x + trans),
        (y - trans, y + trans), (z - trans_z, z + trans_z))
    res = minimize(loss_minimize, initial_guess, args=(two_d_points, weights), bounds=bounds,
                   method='Nelder-Mead',
                   options={'maxiter': 300, 'adaptive': True})

    return res.fun, res.x


def loss_minimize(x, two_d_points, weights):
    internal_loss = loss(x[0], x[1], x[2], x[3], x[4], x[5], two_d_points, weights)
    return internal_loss


def loss(roll: float, pitch: float, yaw: float, x: float, y: float, z: float, two_d_points, weights):
    global iteration, three_d_points
    iteration += 1
    mm = 1e-3
    um = 1e-6
    f = 20 * mm

    r_roll = np.array([[1, 0, 0],
                       [0, math.cos(roll / 180 * math.pi), -math.sin(roll / 180 * math.pi)],
                       [0, math.sin(roll / 180 * math.pi), math.cos(roll / 180 * math.pi)]])

    r_pitch = np.array([[math.cos(pitch / 180 * math.pi), 0, math.sin(pitch / 180 * math.pi)],
                        [0, 1, 0],
                        [-math.sin(pitch / 180 * math.pi), 0, math.cos(pitch / 180 * math.pi)]])

    r_yaw = np.array([[math.cos(yaw / 180 * math.pi), -math.sin(yaw / 180 * math.pi), 0],
                      [math.sin(yaw / 180 * math.pi), math.cos(yaw / 180 * math.pi), 0],
                      [0, 0, 1]])

    rotation_matrix = r_yaw @ r_pitch @ r_roll

    # rotate 3d points
    rotated_3d_points = np.dot(rotation_matrix, three_d_points.T)

    # translate 3d points
    translation_matrix = np.array([[x], [y], [z]])  # might be wrong
    rotated_3d_points += translation_matrix

    # world coordinates to image coordinates
    point2d_rotated = np.zeros((2, len(rotated_3d_points.T)))
    point2d_rotated[0] = rotated_3d_points[0] / (22 + rotated_3d_points[2]) * f / (8.6 * um)
    point2d_rotated[1] = (rotated_3d_points[1] / (22 + rotated_3d_points[2])) * f / (8.6 * um)
    point2d_rotated = point2d_rotated.T

    # matching 2d points with 3d points using KDTree
    tree = KDTree(point2d_rotated)

    weights = np.array(sorted(weights, reverse=True))
    two_d_points = two_d_points[(-weights).argsort()]

    # plot_2d(two_d_points, point2d_rotated,
    #         legend=["Found 2d points",
    #                 "Projected 3d points"],
    #         labels=weights)

    # find the closest 3d point for each 2d point
    closest_points = []
    for point in two_d_points:
        closest = tree.query(point)[1]
        closest_points.append(closest)

    distances = np.linalg.norm(point2d_rotated[closest_points] - two_d_points, axis=1)
    closest_points = np.array(closest_points)
    # if closest_points has duplicates, remove them based on distances
    u, c = np.unique(closest_points, return_counts=True)
    dup = u[c > 1]
    for d in dup:
        # get all indices of d
        indices = np.where(closest_points == d)[0]
        # remove the one with the highest distance
        to_delete = indices[np.argmax(distances[indices])]
        closest_points = np.delete(closest_points, to_delete)
        two_d_points = np.delete(two_d_points, to_delete, axis=0)

    point2d_rotated = point2d_rotated[closest_points]

    # plot_2d(two_d_points, point2d_rotated,
    #         legend=["Found 2d points",
    #                 "Projected 3d points"])

    distances = np.linalg.norm(point2d_rotated - two_d_points, axis=1)
    # exp_lambda = 1 / np.mean(distances)
    # upper_bound = np.log(5) / exp_lambda
    # remove points that are outliers based on distance
    upper_bound = np.mean(distances) + 1.7 * np.std(distances)
    two_d_points = two_d_points[distances < upper_bound]
    point2d_rotated = point2d_rotated[distances < upper_bound]

    # weights should be same length as two_d_points
    weights = weights[:two_d_points.shape[0]]

    if len(two_d_points) < 3:
        # print("too few points")  # TODO maybe decrease std?
        return 1000000

    # calculate the weighted squared loss and return
    weighted_squared_loss = np.sum(np.square(two_d_points - point2d_rotated).T * weights)

    data.append([iteration, weighted_squared_loss])
    points.append([two_d_points, point2d_rotated])

    return weighted_squared_loss


# main
if __name__ == "__main__":
    start_time = timeit.default_timer()
    df_init = pd.read_csv("tracker/best_scores.csv")
    mat = scipy.io.loadmat("tracker/all_vertices_mat.mat")

    initial_guess = df_init.iloc[df_init.index[0]].values[1:][2:]
    # set last 3 to 0 to remove translation
    initial_guess[-3:] = 0

    fine_data = []

    for img_number in tqdm(df_init.index):
        data = []
        points = []
        tries = 2  # number of tries to get a good result

        file_name = df_init.iloc[img_number].values[1:][0]

        im = cv.imread("test_images/dynamic_unknowndeg_0to360_5degstep/" + file_name)
        size = im.shape

        image_points = mat['all_vertices'][img_number][0]

        # remove rows where the last column is below 0.1
        image_points = image_points[image_points[:, 2] > 0.1]

        weights = image_points[:, 2]

        image_points = image_points[:, :2]

        image_points[:, 0] += 104

        image_points -= size[1] // 2, size[0] // 2

        # plot_2d(image_points, title=f"original image {img_number}", labels=weights)

        while True:
            internal_loss, new_guess = match_three_d(image_points, weights, initial_guess)
            if internal_loss < 5 or tries == 0:
                break
            else:
                tries -= 1
                initial_guess = new_guess

        fine_data.append([img_number, iteration, internal_loss, new_guess[0], new_guess[1], new_guess[2], new_guess[3],
                          new_guess[4],
                          new_guess[5]])

        initial_guess = new_guess
        iteration = 0

        if img_number == -1:
            for i, row in enumerate(points):
                if i == 0 or i == len(points) - 1:
                    time.sleep(0.5)
                    plot_2d(row[0], row[1], title=f"iteration {data[i][0]} loss {data[i][1]:.2f}",
                            legend=["Found 2d points",
                                    "Projected 3d points"])
                    time.sleep(0.5)

    end_time = timeit.default_timer()
    print(f"Time: {end_time - start_time:.2f}s")

    df_fine = pd.DataFrame(fine_data,
                           columns=["img_number", 'iterations', "loss", "roll", "pitch", "yaw", "x", "y", "z"])
    df_fine.to_csv("tracker/fine_scores.csv", index=False)
