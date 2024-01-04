class Rotation:
    """
    Rotation of an object in 3D space
    """
    roll: float = None  # Rotation around the x-axis
    pitch: float = None  # Rotation around the y-axis
    yaw: float = None  # Rotation around the z-axis

    def __init__(self, roll: float = None, pitch: float = None, yaw: float = None):
        if roll is not None:
            self.roll = float(roll)
        if pitch is not None:
            self.pitch = float(pitch)
        if yaw is not None:
            self.yaw = float(yaw)

    def __str__(self):
        # 2 decimal places and roll, pitch and yaw can be None and should not be printed
        if self.roll is None:
            self.roll = 0
        if self.pitch is None:
            self.pitch = 0
        if self.yaw is None:
            self.yaw = 0

        return f"roll: {self.roll:.2f}, pitch/theta/X: {self.pitch:.2f}, yaw/phi/Y: {self.yaw:.2f}"

    # define sort order to sort ascending by pitch and then yaw can be None
    def __lt__(self, other):
        if self.pitch < other.pitch:
            return True
        else:
            return False

    def set_roll(self, roll: float):
        self.roll = roll

    def add_roll(self, roll: float):
        self.roll += roll
        self.roll %= 360

    def set_pitch(self, pitch: float):
        self.pitch = pitch

    def set_yaw(self, yaw: float):
        self.yaw = yaw


class Translation:
    """
    Translation of an object in 3D space
    """
    x: float = None
    y: float = None
    z: float = None

    def __init__(self, x: float or list = None, y: float = None, z: float = None):
        # if x is a list, unpack it
        if isinstance(x, list):
            x, y, z = x
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


class Pose:
    """
    Pose of an object in 3D space
    """
    translation: Translation = None
    rotation: Rotation = None
    channel: int = None

    def __init__(self, translation: Translation, rotation: Rotation, channel: int = None):
        self.translation = translation
        self.rotation = rotation
        if channel is not None:
            self.channel = channel

    def __str__(self):
        # self.channel can be None
        if self.channel is None:
            self.channel = 0
        return f"channel: {self.channel}, translation: {self.translation}, rotation: {self.rotation}"

    def __lt__(self, other):
        if self.channel < other.channel:
            return True
        else:
            return False

    def get_translation(self):
        return self.translation

    def get_rotation(self):
        return self.rotation

    def set_roll(self, roll: float):
        self.rotation.set_roll(roll)

    def get_channel(self):
        return self.channel

    def set_rotation(self, rotation: Rotation):
        self.rotation = rotation
