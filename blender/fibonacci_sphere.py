# Plots the fibonnaci sphere

import math
import matplotlib.pyplot as plt


def fibonacci_sphere(samples):
    points = []
    phi = math.pi * (math.sqrt(5.) - 1.)  # golden angle in radians

    for i in range(samples):
        y = 1 - (i / float(samples - 1)) * 2  # y goes from 1 to -1
        radius = math.sqrt(1 - y * y)  # radius at y

        theta = phi * i  # golden angle increment

        x = math.cos(theta) * radius
        z = math.sin(theta) * radius

        points.append((x, y, z))

    return points

fig = plt.figure(figsize=(12, 12))
ax = fig.add_subplot(projection='3d')

# plot the points
for x, y, z in fibonacci_sphere(200):
    ax.scatter(x, y, z, color='b')

plt.show()
