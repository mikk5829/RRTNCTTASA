# evenly distribute 360 degrees into 10 points
import numpy as np

angle = np.linspace(0, 360, 10)
angle2 = np.linspace(0, 360, 10)

# make combinations of the two arrays
n = 11
angle3 = np.array(np.meshgrid(np.linspace(0, np.pi * 2, n), np.linspace(0, np.pi * 2, n))).T.reshape(-1, 2)

# mod np.pi * 2
angle3 = np.mod(angle3, np.pi * 2)
# make unique combinations
angle3 = np.unique(angle3, axis=0)

print(len(angle3))
