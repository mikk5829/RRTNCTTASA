from dependency_injector import containers, providers

from image.contour_service import ContourService
from image.image_service import ImageService
from image.pose_map_service import PoseMapService
from tracker import Tracker


class Container(containers.DeclarativeContainer):
    """
    Container for dependency injection
    """
    config = providers.Configuration()

    pose_map_importer = providers.Singleton(PoseMapService, path_to_model_images=config.path_to_model_images,
                                            model_name=config.model_name)

    contour_service = providers.Singleton(ContourService, image_path=config.image_path, model_name=config.model_name)

    image_service = providers.Singleton(ImageService, image_path=config.image_path, model_name=config.model_name)

    tracker = providers.Singleton(Tracker, pose_map_importer=pose_map_importer, image_path=config.image_path,
                                  image_service=image_service)
