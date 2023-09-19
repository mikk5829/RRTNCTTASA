from models.moments import Coordinates, Moments


class Object:
    coordinates: Coordinates = None
    moments: Moments = None
    contours = None
    aligned_image = None

    def __init__(self, moments: Moments, contours):
        self.moments = moments
        self.coordinates = moments.get_coordinates()
        self.contours = contours

    def set_aligned_image(self, image):
        self.aligned_image = image

    def __str__(self):
        return f"Coordinates: {self.coordinates}"
