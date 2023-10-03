from image.image_service import ImageService
from image.pose_map_service import PoseMapService
from models.pose import Translation
from service.service_interface import IService
import cv2 as cv


class ContourService(IService):
    image_path = None
    model_name = None
    image_service: ImageService = None
    object_service = None
    pose_map_service: PoseMapService = None

    def __init__(self, config, image_service, object_service, pose_map_service):
        super().__init__(config)
        self.image_service = image_service
        self.object_service = object_service
        self.pose_map_service = pose_map_service

    # TODO Move to Object class
    def simplify_contours(self):
        tracked_object = self.object_service.get_object()
        contours = tracked_object.get_relative_contour()
        # using douglas peucker algorithm reduce the amount of lines
        epsilon = 0.003 * cv.arcLength(contours, True)
        approx = cv.approxPolyDP(contours, epsilon, True)
        tracked_object.set_contour(approx)
        self.object_service.set_object(approx)

    def get_best_match(self) -> (Translation, cv.UMat):
        tracked_object = self.object_service.get_object()
        contours = tracked_object.get_relative_contour()

        # match the contours to the model
        pose_map = self.pose_map_service.get_pose_map()
        best_score = 1  # initialize with the worst score
        found_translation = None
        found_model_contour = None

        for translation, model_contour in pose_map.items():
            local_score = cv.matchShapes(contours, model_contour, 1, 0.0)
            if best_score > local_score:  # if the score is better than the previous best score
                best_score = local_score  # set the new best score
                found_translation = translation  # set the new best translation
                found_model_contour = model_contour  # set the new best model contour

        return found_translation, found_model_contour
