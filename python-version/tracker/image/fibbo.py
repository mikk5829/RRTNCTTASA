import numpy as np

num_points = 10
ga = (3 - np.sqrt(5)) * np.pi  # golden angle

# Create a list of golden angle increments along tha range of number of points
theta = ga * np.arange(num_points)

# Z is a split into a range of -1 to 1 in order to create a unit circle
z = np.linspace(1 / num_points - 1, 1 - 1 / num_points, num_points)

# a list of the radii at each height step of the unit circle
radius = np.sqrt(1 - z * z)

# Determine where xy fall on the sphere, given the azimuthal and polar angles
y = radius * np.sin(theta)
x = radius * np.cos(theta)

theta_rad = theta % (2 * np.pi)
theta_deg = np.rad2deg(theta_rad)
# convert -1 to -np.pi and 1 to np.pi
phi_rad = z * np.pi
phi_deg = z * 90

data = np.array([theta_rad, phi_rad, theta_deg, phi_deg]).T

distMat = np.zeros((num_points, num_points))
for i in range(num_points):
    diff = np.transpose(np.array([x, y, z])) - np.array([x[i], y[i], z[i]])
    distMat[:, i] = np.linalg.norm(diff, axis=1)

furthest_idx = np.argmax(distMat, axis=0)

# furthest_idx f string integer
print(f"furthest_idx: {furthest_idx}")

# add furthest_idx to data
data = np.concatenate((data, furthest_idx.reshape(-1, 1)), axis=1)

print(data)

# for frame, (tr, pr, td, pd) in enumerate(data):
#     print(tr, pr, td, pd)
