class Rotation:
    """
    Rotation of an object in 3D space
    """
    roll: float = None  # Rotation around the x-axis
    pitch: float = None  # Rotation around the y-axis
    yaw: float = None  # Rotation around the z-axis

    def __init__(self, roll: float = None, pitch: float = None, yaw: float = None):
        self.roll = roll
        self.pitch = pitch
        self.yaw = yaw

    def __str__(self):
        return f"roll: {self.roll}, pitch: {self.pitch}, yaw: {self.yaw}"


class Translation:
    """
    Translation of an object in 3D space
    """
    x: float = None
    y: float = None
    z: float = None

    def __init__(self, x: float = None, y: float = None, z: float = None):
        self.x = x
        self.y = y
        self.z = z

    def __str__(self):
        return f"x: {self.x}, y: {self.y}, z: {self.z}"


class Pose(Translation, Rotation):
    """
    Pose of an object in 3D space
    """

    def __init__(self, translation: Translation = None, rotation: Rotation = None):
        if translation is not None:
            Translation.__init__(self, translation.x, translation.y, translation.z)
        if rotation is not None:
            Rotation.__init__(self, rotation.roll, rotation.pitch, rotation.yaw)

    def set_translation(self, x: float, y: float, z: float):
        """
        Set the location of the object
        """
        Translation.__init__(self, x, y, z)

    def set_rotation(self, roll: float, pitch: float, yaw: float):
        """
        Set the rotation of the object
        :param roll: The rotation around the x-axis
        :param pitch: The rotation around the y-axis
        :param yaw: The rotation around the z-axis
        """
        Rotation.__init__(self, roll, pitch, yaw)
