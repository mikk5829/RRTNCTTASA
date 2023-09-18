class Pose:
    """
    Pose of an object in 3D space
    """
    x: float = None
    y: float = None
    z: float = None
    roll: float = None  # Rotation around the x-axis
    pitch: float = None  # Rotation around the y-axis
    yaw: float = None  # Rotation around the z-axis

    def __init__(self, x: float = None, y: float = None, z: float = None, roll: float = None, pitch: float = None,
                 yaw: float = None):
        self.x = x
        self.y = y
        self.z = z
        self.roll = roll
        self.pitch = pitch
        self.yaw = yaw

    def __str__(self):
        return f"x: {self.x}, y: {self.y}, z: {self.z}, roll: {self.roll}, pitch: {self.pitch}, yaw: {self.yaw}"

    def set_location(self, x: float, y: float, z: float):
        """
        Set the location of the object
        """
        self.x = x
        self.y = y
        self.z = z

    def set_rotation(self, roll: float, pitch: float, yaw: float):
        """
        Set the rotation of the object
        :param roll: The rotation around the x-axis
        :param pitch: The rotation around the y-axis
        :param yaw: The rotation around the z-axis
        """
        self.roll = roll
        self.pitch = pitch
        self.yaw = yaw
