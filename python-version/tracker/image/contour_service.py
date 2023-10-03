from matplotlib import pyplot as plt

from image.image_service import ImageService
from image.object_service import ObjectService
from image.pose_map_service import PoseMapService
from models.pose import Translation
from service.service_interface import IService
import cv2 as cv


class ContourService(IService):
    image_path = None
    model_name = None
    __image_service: ImageService = None
    __object_service: ObjectService = None
    __pose_map_service: PoseMapService = None

    def __init__(self, config, image_service, object_service, pose_map_service):
        super().__init__(config)
        self.__image_service = image_service
        self.__object_service = object_service
        self.__pose_map_service = pose_map_service

    def get_best_match(self) -> (Translation, cv.UMat):
        tracked_object = self.__object_service.get_object()
        contours = tracked_object.get_relative_contour()

        # match the contours to the model
        pose_map = self.__pose_map_service.get_pose_map()
        best_score = 1  # initialize with the worst score
        found_translation = None
        found_model_contour = None

        for translation, model_contour in pose_map.items():
            local_score = cv.matchShapes(contours, model_contour, 1, 0.0)
            if best_score > local_score:  # if the score is better than the previous best score
                # plot the best match
                plt.plot(contours[:, 0, 0], contours[:, 0, 1], "r")
                plt.plot(model_contour[:, 0, 0], model_contour[:, 0, 1], "b")
                plt.title(f"Score: {local_score}")
                plt.show()
                best_score = local_score  # set the new best score
                found_translation = translation  # set the new best translation
                found_model_contour = model_contour  # set the new best model contour

        return found_translation, found_model_contour
