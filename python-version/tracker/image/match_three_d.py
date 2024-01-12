import time

import matplotlib.pyplot as plt
import numpy as np
import math
import cv2 as cv
import pandas as pd
from scipy.optimize import minimize

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
im = cv.imread("test_images/dynamic_unknowndeg_0to360_5degstep/A240105_15362816.png")


def plot_3d(points):
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    # change the view such that y is down, x is right and z is in the screen
    ax.view_init(azim=-90, elev=-90)
    ax.scatter(points[:, 0], points[:, 1], points[:, 2])
    for i, txt in enumerate(labels):
        ax.text(points[i][0], points[i][1], points[i][2], txt)
    # axis should have the same range -1 to 1
    ax.set_xlim3d(-0.5, 0.5)
    ax.set_ylim3d(-0.5, 0.5)
    ax.set_zlim3d(-0.5, 0.5)
    ax.set_xlabel('X Label')
    ax.set_ylabel('Y Label')
    ax.set_zlabel('Z Label')
    plt.show()


def plot_2d(points, extra_points=None, title=""):
    plt.imshow(im)
    plt.title(title)
    for i, txt in enumerate(labels):
        plt.plot(points[i][0], points[i][1], 'ro')
        # annotate with red text
        plt.annotate(txt, (points[i][0], points[i][1]), color='red')
        if extra_points is not None:
            plt.plot(extra_points[i][0], extra_points[i][1], 'bo')
            plt.annotate(txt, (extra_points[i][0], extra_points[i][1]), color='blue')
    plt.savefig(f"test_images/dynamic_unknowndeg_0to360_5degstep/minimize/{title}.png")
    plt.show()


three_d_points = np.array([
    [-0.01, -0.34, 0.18],  # Top right solar panel
    [-0.02, -0.13, 0.18],  # Top left solar panel
    [-0.01, -0.35, -0.12],  # Bottom right solar panel
    [-0.02, -0.13, -0.12],  # Bottom left solar panel
    [-0.04, 0.15, -0.1],  # Bottom left satellite
    [-0.04, 0.15, 0.1],  # Top left satellite
])

data = []
points = []
iteration = 0


# make a function that takes in the 3d points, the 2d points and the initial guess of the pose
# and returns the best pose
def match_three_d(two_d_points, initial_guess):
    bounds = ((None, None), (None, None), (None, None), (0, 360), (0, 360), (0, 360))
    res = minimize(loss_minimize, initial_guess, args=(two_d_points))
    rotation = Rotation(res.x[0], res.x[1], res.x[2])
    translation = Translation(res.x[3], res.x[4], res.x[5])
    print(rotation)
    print(translation)


def loss_minimize(x, two_d_points):
    return loss(x[0], x[1], x[2], x[3], x[4], x[5], two_d_points)


def loss(roll, pitch, yaw, x, y, z, two_d_points):
    global iteration
    iteration += 1
    mm = 1e-3
    um = 1e-6
    f = 20 * mm

    # Define the camera matrix
    # camera_matrix = np.array([[f, 0, cx], [0, f, cy], [0, 0, 1]])

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
    translation_matrix = np.array([[x], [y], [z]])
    rotated_3d_points += translation_matrix

    # world coordinates to camera coordinates
    point2d_rotated = np.zeros((2, len(two_d_points)))
    point2d_rotated[0] = rotated_3d_points[0] / (1 - rotated_3d_points[2]) * f / (8.6 * um)  # pixel size
    point2d_rotated[1] = rotated_3d_points[1] / (1 - rotated_3d_points[2]) * f / (8.6 * um)
    point2d_rotated = point2d_rotated.T

    # calculate the squared loss and return
    squared_loss = np.sum(np.square(two_d_points - point2d_rotated))

    # if iteration % 100000 == 0:
    data.append([iteration, squared_loss])
    points.append([two_d_points, point2d_rotated])
    # plot_3d(rotated_3d_points.T)
    # plot_2d(two_d_points, point2d_rotated)
    # print(squared_loss)

    return squared_loss


# main
if __name__ == "__main__":
    # im = cv.imread("test_images/testpnp2.png")
    # size = im.shape
    # image_points = np.array([
    #     [420, size[0] - 770],  # Top right solar panel
    #     [470, size[0] - 570],  # Top left solar panel
    #     [806, size[0] - 856],  # Bottom right solar panel
    #     [915, size[0] - 695],  # Bottom left solar panel
    #     [1100, size[0] - 367],  # Bottom left satellite
    #     [780, size[0] - 240],  # Top left satellite
    # ], dtype="double")

    size = im.shape
    image_points = np.array([
        [542, size[0] - 506],  # Top right solar panel
        [408, size[0] - 417],  # Top left solar panel
        [658, size[0] - 336],  # Bottom right solar panel
        [527, size[0] - 252],  # Bottom left solar panel
        [354, size[0] - 159],  # Bottom left satellite
        [275, size[0] - 265],  # Top left satellite
    ], dtype="double")

    # plot the points on the image
    for i, txt in enumerate(labels):
        plt.plot(image_points[i][0], image_points[i][1], 'ro')
        # annotate with red text
        plt.annotate(txt, (image_points[i][0], image_points[i][1]), color='red')
    plt.imshow(im)
    plt.show()

    # plot three_d_points in 3d
    # plot_3d(three_d_points)

    # vores roll er rundt om z aksen, pitch er rundt om x aksen og yaw er rundt om y aksen
    match_three_d(image_points, [137.508, -88.65, 282, 0, 0, 0])

    # for row in data:
    #     print(row)

    for i, row in enumerate(points):
        if (i + 1) % 30 == 0:
            time.sleep(1)
            plot_2d(row[0], row[1], title=f"iteration {data[i][0]} loss {data[i][1]:.2f}")
