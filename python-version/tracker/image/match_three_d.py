import cProfile
import io
import itertools
import pstats
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


def prof_to_csv(prof: cProfile.Profile):
    out_stream = io.StringIO()
    pstats.Stats(prof, stream=out_stream).strip_dirs().print_stats()
    result = out_stream.getvalue()
    # chop off header lines
    result = 'ncalls' + result.split('ncalls')[-1]
    lines = [','.join(line.rstrip().split(None, 5)) for line in result.split('\n')]
    return '\n'.join(lines)


labels = ["TR", "TL", "BR", "BL", "BLS", "TLS"]


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
    # remove 104 from left and 68 from right of im

    plt.imshow(im, extent=[-size[1] // 2, size[1] // 2, size[0] // 2, -size[0] // 2])

    # plt.imshow(im)
    # plt.title(title)
    # scatter plot points
    # remove white space
    plt.gca().set_axis_off()
    plt.scatter(points[:, 0], points[:, 1], c='r')
    # remove the axis
    plt.axis('off')
    # for point in points:
    #     plt.plot(point[0], point[1], 'ro')
    if extra_points is not None:
        plt.scatter(extra_points[:, 0], extra_points[:, 1], c='y')
        # for point in extra_points:
        #     plt.plot(point[0], point[1], 'bo')

    if labels is not None:
        for i, txt in enumerate(labels):
            plt.text(points[i][0], points[i][1], f"{txt:.2f}", c='r')
    # if legend:
    # plt.legend(legend)
    plt.savefig(f"test_images/dynamic_unknowndeg_0to360_5degstep/minimize/{title}.png", bbox_inches='tight',
                pad_inches=0)
    plt.show()


# points in meters around the center of mass
# TR = [-0.080903, -2.30455, 1.19694] m
# BR = [-0.080903, -2.30455, -0.783056] m
# TL = [-0.157453, -0.843874, 1.19694] m
# BL = [-0.157453, -0.843874, -0.783056] m
# BLsat = [-0.244135, 1.03251, -0.65297] m
three_d_points = np.array([
    [-0.080903, -2.30455, 1.19694],  # Top right solar panel
    [-0.157453, -0.843874, 1.19694],  # Top left solar panel
    [-0.080903, -2.30455, -0.783056],  # Bottom right solar panel
    [-0.157453, -0.843874, -0.783056],  # Bottom left solar panel
    [-0.244135, 1.03251, -0.65297],  # Bottom left satellite
])

# three_d_points = np.array([
#     [-0.08 - 0.1069, -2.30333333, 1.19666667],  # Top right solar panel
#     [-0.15666667 - 0.1069, -0.84333333, 1.19666667],  # Top left solar panel
#     [-0.08 - 0.1069, -2.30333333, -0.78333333],  # Bottom right solar panel
#     [-0.15666667 - 0.1069, -0.84333333, -0.78333333],  # Bottom left solar panel
#     [-0.24333333 - 0.1069, 1.03, -0.65333333],  # Bottom left satellite
#     # [-0.24666667, 1.01333333, 0.69333333],  # Top left satellite
#     # [0.7566, 1.03, -0.65333333],
# ])

# three_d_points -= [0.2799 - 0.1329 - 0.1785]

data = []
points = []
all_distances = []
iteration = 0


def match_three_d(two_d_points, weights, initial_guess, bounds):
    res = minimize(loss_minimize, initial_guess, args=(two_d_points, weights), bounds=bounds,
                   method='L-BFGS-B')

    return res.fun, res.x


def loss_minimize(x, two_d_points, weights):
    internal_loss = loss(x[0], x[1], x[2], x[3], x[4], x[5], two_d_points, weights)
    return internal_loss


def loss(roll: float, pitch: float, yaw: float, x: float, y: float, z: float, two_d_points, weights):
    global iteration, three_d_points
    # iteration += 1
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
    point2d_rotated[1] = rotated_3d_points[1] / (22 + rotated_3d_points[2]) * f / (8.3 * um)

    point2d_rotated = point2d_rotated.T

    # kappa = 2.3e-8
    # r = (point2d_rotated[:, 0] ** 2 + point2d_rotated[:, 1] ** 2)
    #
    # point2d_rotated[:, 0] = point2d_rotated[:, 0] / (1 + kappa * r)
    # point2d_rotated[:, 1] = point2d_rotated[:, 1] / (1 + kappa * r)

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
        # find the closest point
        closest = tree.query(point)[1]
        if closest in closest_points:
            closest = tree.query(point, k=2)[1][1]
        if closest in closest_points:
            closest = tree.query(point, k=3)[1][2]
        closest_points.append(closest)

    distances = np.linalg.norm(point2d_rotated[closest_points] - two_d_points, axis=1)
    closest_points = np.array(closest_points)
    # if closest_points has duplicates, remove them based on distances
    to_delete = np.array([], dtype=int)
    u, c = np.unique(closest_points, return_counts=True)
    dup = u[c > 1]
    for d in dup:
        # get all indices of d
        indices = np.where(closest_points == d)[0]
        # keep the one with the lowest  distance
        to_keep = indices[np.argmin(distances[indices])]
        to_delete = np.append(to_delete, indices[np.where(indices != to_keep)[0]])

    # selecte the points that are not in to_delete
    mask = np.full(len(closest_points), True, dtype=bool)
    mask[to_delete] = False
    closest_points = closest_points[mask]
    two_d_points = two_d_points[mask]
    point2d_rotated = point2d_rotated[closest_points]

    # plot_2d(two_d_points, point2d_rotated,
    #         legend=["Found 2d points",
    #                 "Projected 3d points"])

    distances = np.linalg.norm(point2d_rotated - two_d_points, axis=1)
    exp_lambda = 1 / np.mean(distances)
    upper_bound = np.log(5) / exp_lambda
    # remove points that are outliers based on distance
    # upper_bound = np.mean(distances) + 2 * np.std(distances)

    two_d_points = two_d_points[distances < upper_bound]
    point2d_rotated = point2d_rotated[distances < upper_bound]

    # weights should be same length as two_d_points
    weights = weights[:two_d_points.shape[0]]

    if len(two_d_points) < 3:
        # print("too few points")  # TODO maybe decrease std?
        return 1000000

    # calculate the weighted squared loss and return

    diff_l2 = np.linalg.norm(two_d_points - point2d_rotated, axis=1)
    weighted_squared_loss = np.average(
        diff_l2 ** 2, axis=0, weights=weights)

    data.append([iteration, weighted_squared_loss])
    points.append([two_d_points, point2d_rotated])

    return weighted_squared_loss


# main
if __name__ == "__main__":
    pr = cProfile.Profile()
    pr.enable()
    folder = "test_images/replica_of_dynamic_unknowndeg_0to360_5degstep/"
    start_time = timeit.default_timer()
    df_init = pd.read_csv(folder + "best_scores.csv")
    # remove last row
    # df_init = df_init[:-1]
    # df_init = pd.concat([df_init, df_init], ignore_index=True)
    suffix = "_synthetic"
    mat = scipy.io.loadmat(folder + "all_vertices" + suffix + ".mat")
    # remove last row
    # mat['all_vertices'] = mat['all_vertices'][:-1]
    # mat['all_vertices'] = np.concatenate((mat['all_vertices'], mat['all_vertices']), axis=0)

    initial_guess = df_init.iloc[df_init.index[0]].values[1:][2:]
    initial_guess = [168.68, -72.54, -82.47, 0, 0, 0]
    # set last 3 to 0 to remove translation
    # initial_guess[-1:] -= 22

    fine_data = []
    guess_data = []

    deg_roll_sma = 0
    deg_pitch_sma = 0
    deg_yaw_sma = 0
    trans_x_sma = 0
    trans_y_sma = 0
    trans_z_sma = 0

    deg_roll_std = 8
    deg_pitch_std = 4
    deg_yaw_std = 4
    trans_x_std = 0.1
    trans_y_std = 0.1
    trans_z_std = 0.1

    trial_multiplier = 8  # high value means more uncertainty

    for img_number in tqdm(df_init.index):
        data = []
        points = []
        tries = 3  # number of tries to get a good result

        file_name = df_init.iloc[img_number].values[1:][0]
        # file_name = df_init.iloc[img_number].values[0:][0]

        im = cv.imread(folder + str(file_name))
        size = im.shape
        width = size[1]
        height = size[0]

        principal_point = (width // 2, height // 2)

        image_points = mat['all_vertices'][img_number][0]

        # remove rows where the weight column is below 0.11
        # image_points = image_points[image_points[:, 2] > 0.11]

        weights = image_points[:, 2]

        image_points = image_points[:, :2]
        # minus one in the y axis
        # image_points[:, 1] -= 0.54

        # image_points[:, 0] += 104

        image_points -= principal_point

        # if tries < 3:
        # plot_2d(image_points, title=f"original image {img_number}", labels=weights)

        roll, pitch, yaw, x, y, z = initial_guess
        roll += deg_roll_sma
        pitch += deg_pitch_sma
        yaw += deg_yaw_sma
        x += trans_x_sma
        y += trans_y_sma
        z += trans_z_sma
        initial_guess = [roll, pitch, yaw, x, y, z]

        deg_roll = deg_roll_std * trial_multiplier
        deg_pitch = deg_pitch_std * trial_multiplier
        deg_yaw = deg_yaw_std * trial_multiplier
        trans_x = trans_x_std * trial_multiplier
        trans_y = trans_y_std * trial_multiplier
        trans_z = trans_z_std * trial_multiplier

        bounds = (
            (roll - deg_roll, roll + deg_roll), (pitch - deg_pitch, pitch + deg_pitch), (yaw - deg_yaw, yaw + deg_yaw),
            (x - trans_x, x + trans_x),
            (y - trans_y, y + trans_y), (z - trans_z, z + trans_z))

        while True:
            internal_loss, new_guess = match_three_d(image_points, weights, initial_guess, bounds)
            if internal_loss < 10:  # good result
                initial_guess = new_guess
                trial_multiplier = 1  # reset trial multiplier
                break
            elif tries == 0:  # no more tries
                break  # no more tries
            else:  # bad result
                tries -= 1
                if trial_multiplier < 5:
                    trial_multiplier += 1  # increase trial multiplier to increase uncertainty

        guess_data.append(
            [img_number, roll, pitch, yaw, x, y, z, bounds[0][0], bounds[0][1], bounds[1][0], bounds[1][1],
             bounds[2][0], bounds[2][1], bounds[3][0], bounds[3][1], bounds[4][0], bounds[4][1],
             bounds[5][0], bounds[5][1]])

        fine_data.append([img_number, iteration, internal_loss, new_guess[0], new_guess[1], new_guess[2], new_guess[3],
                          new_guess[4],
                          new_guess[5]])

        n_ma = 10  # number of points to use for moving average
        if len(fine_data) > n_ma:
            # simple moving average
            mean_diff = np.mean(np.diff(np.array(fine_data[-n_ma:]), axis=0), axis=0)
            std_diff = np.std(np.diff(np.array(fine_data[-n_ma:]), axis=0), axis=0)
            deg_roll_sma = mean_diff[3]
            deg_pitch_sma = mean_diff[4]
            deg_yaw_sma = mean_diff[5]
            trans_x_sma = mean_diff[6]
            trans_y_sma = mean_diff[7]
            trans_z_sma = mean_diff[8]

            deg_roll_std = std_diff[3] * 2
            deg_pitch_std = std_diff[4] * 2
            deg_yaw_std = std_diff[5] * 2
            trans_x_std = std_diff[6] * 2
            trans_y_std = std_diff[7] * 2
            trans_z_std = std_diff[8] * 2

        iteration = 0

        if False:
            for i, row in enumerate(points):
                if i == 0 or i == len(points) - 1:
                    time.sleep(0.5)
                    # print("Trial multiplier: " + str(trial_multiplier))
                    plot_2d(row[0], row[1], title=f"iteration {i} loss {data[i][1]:.2f}, image {img_number}",
                            legend=["Found 2D points",
                                    "Projected 3D points"])
                    time.sleep(0.5)

    end_time = timeit.default_timer()
    print(f"Time: {end_time - start_time:.2f}s")
    pr.disable()
    csv = prof_to_csv(pr)
    with open("prof_fine.csv", 'w+') as f:
        f.write(csv)

    df_fine = pd.DataFrame(fine_data,
                           columns=["img_number", 'iterations', "loss", "roll", "pitch", "yaw", "x", "y", "z"])
    df_fine.to_csv(folder + "fine_scores" + suffix + '.csv', index=False)

    df_guess = pd.DataFrame(guess_data,
                            columns=["img_number", "roll", "pitch", "yaw", "x", "y", "z", "roll_min", "roll_max",
                                     "pitch_min", "pitch_max", "yaw_min", "yaw_max", "x_min", "x_max", "y_min",
                                     "y_max", "z_min", "z_max"])
    df_guess.to_csv(folder + "guesses" + suffix + '.csv', index=False)
