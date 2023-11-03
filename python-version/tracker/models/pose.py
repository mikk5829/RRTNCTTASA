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
        # 2 decimal places and roll, pitch and yaw can be None
        if self.roll is None:
            self.roll = 0
        if self.pitch is None:
            self.pitch = 0
        if self.yaw is None:
            self.yaw = 0

        return f"roll: {self.roll:.2f}, pitch: {self.pitch:.2f}, yaw: {self.yaw:.2f}"

    # define sort order to sort ascending by pitch and then yaw can be None
    def __lt__(self, other):
        if self.pitch < other.pitch:
            return True
        else:
            return False


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
        if self.x is None:
            self.x = 0
        if self.y is None:
            self.y = 0
        if self.z is None:
            self.z = 0
        return f"x: {self.x:.2f}, y: {self.y:.2f}, z: {self.z:.2f}"


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

    def __lt__(self, other):
        Rotation.__lt__(self, other)

    def __str__(self):
        return f"translation: {Translation.__str__(self)}, rotation: {Rotation.__str__(self)}"
