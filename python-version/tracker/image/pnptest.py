import cv2 as cv
import numpy as np

from models.pose import cv

# Read Image

# 2D image points. If you change the image, you need to change vector
# TODO har det betydning hvilken rækkefølge de er i?
# TODO har det betydning om 3D shaper = 2D shape?
# testpnp.png
# im = cv.imread("test_images/testpnp.png");
# size = im.shape
# image_points = np.array([
#     (1330, size[0] - 733),  # Top right solar panel
#     (1097, size[0] - 732),  # Top left solar panel
#     (1332, size[0] - 411),  # Bottom right solar panel
#     (1098, size[0] - 411),  # Bottom left solar panel
#     (789, size[0] - 428),  # Bottom left satellite
#     (798, size[0] - 629),  # Top left satellite
# ], dtype="double")
# setup 4/A231108_13371599.png
# im = cv.imread("test_images/setup 4/A231108_13371599.png");
# size = im.shape
# image_points = np.array([
#     (200, size[0] - 440),  # Top right solar panel
#     (248, size[0] - 294),  # Top left solar panel
#     (427, size[0] - 491),  # Bottom right solar panel
#     (481, size[0] - 351),  # Bottom left solar panel
#     (535, size[0] - 146),  # Bottom left satellite
#     (374, size[0] - 105),  # Top left satellite
# ], dtype="double")

im = cv.imread("test_images/testpnp2.png");
size = im.shape
image_points = np.array([
    (420, size[0] - 770),  # Top right solar panel
    (470, size[0] - 570),  # Top left solar panel
    (806, size[0] - 856),  # Bottom right solar panel
    (915, size[0] - 695),  # Bottom left solar panel
    (1100, size[0] - 367),  # Bottom left satellite
    (780, size[0] - 240),  # Top left satellite
], dtype="double")

image_points_names = ["TR", "TL", "BR", "BL", "BLS", "TLS"]

# 3D model points.
model_points = np.array([
    (0.35, -0.18, -0.01),  # Top right solar panel
    (0.13, -0.18, -0.02),  # Top left solar panel
    (0.35, 0.12, -0.01),  # Bottom right solar panel
    (0.13, 0.12, -0.02),  # Bottom left solar panel
    (-0.15, 0.09, -0.04),  # Bottom left satellite
    (-0.15, -0.08, -0.04),  # Top left satellite
])

# Camera internals
f = 55  # focal length in mm TODO
sx = 22.3  # sensor size
sy = 14.9  # sensor size
width = size[1]
height = size[0]
fx = width * f / sx
fy = height * f / sy

focal_length = size[1]
center = (size[1] / 2, size[0] / 2)
camera_matrix = np.array(
    [[focal_length, 0, center[0]],
     [0, focal_length, center[1]],
     [0, 0, 1]], dtype="double"
)

print("Camera Matrix :\n {0}".format(camera_matrix))

dist_coeffs = np.zeros((4, 1))  # Assuming no lens distortion
(success, rotation_vector, translation_vector) = cv.solvePnP(model_points, image_points, camera_matrix, dist_coeffs,
                                                             flags=cv.SOLVEPNP_ITERATIVE)

print("Rotation Vector:\n {0}".format(rotation_vector))
# convert rotation vector to degrees
rotation_vector_deg = np.rad2deg(rotation_vector)
rotation = Rotation(rotation_vector_deg[0][0], rotation_vector_deg[1][0], rotation_vector_deg[2][0])
print(rotation)
# display the rotation with pitch, yaw and roll
print("Translation Vector:\n {0}".format(translation_vector))

# make rotation_vector to rotation matrix
rotation_matrix = cv.Rodrigues(rotation_vector)[0]

# rotation-translation matrix:
rot_trans_matrix = np.hstack((rotation_matrix, translation_vector))


def backproject(point):
    point = np.array(point)
    point = np.append(point, 1)
    point = np.transpose(point)
    threed_vector = camera_matrix.dot(rot_trans_matrix).dot(point)

    return threed_vector[0] / threed_vector[2], threed_vector[1] / threed_vector[2]


vector = []
for point in model_points:
    vector.append(backproject(point))

print("Backprojected points:\n {0}".format(vector))

# cam_worl_pos = - inverse(R) * tvec
cam_worl_pos = np.matmul(- np.transpose(rotation_matrix), translation_vector)
print("Camera World Position:\n {0}".format(cam_worl_pos))

translation_vector_bcs = np.matmul(rotation_matrix, translation_vector)
print("Translation Vector BCS:\n {0}".format(translation_vector_bcs))

# Project a 3D point (0, 0, 1000.0) onto the image plane.
# We use this to draw a line sticking out of the nose

# (nose_end_point2D, jacobian) = cv.projectPoints(np.array([(0.0, 0.0, 1000.0)]), rotation_vector, translation_vector,
#                                                 camera_matrix, dist_coeffs)

for i, p in enumerate(image_points):
    cv.circle(im, (int(p[0]), int(p[1])), 3, (255, 255, 255), 5)
    # draw a letter next to the point
    cv.putText(im, image_points_names[i], (int(p[0]) + 10, int(p[1]) + 10), cv.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255),
               2)

p1 = (int(image_points[0][0]), int(image_points[0][1]))
# p2 = (int(nose_end_point2D[0][0][0]), int(nose_end_point2D[0][0][1]))

# cv.line(im, p1, p2, (255, 0, 0), 2)

# Display image
cv.imshow("Output", im)
cv.waitKey(0)
