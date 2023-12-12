from dependency_injector import containers, providers

from image.contour_service import ContourService
from image.image_service import ImageService
from image.object_service import ObjectService
from image.pose_map_service import PoseMapService
from service.service_interface import IService
from tracker import Tracker


class Container(containers.DeclarativeContainer):
    """
    Container for dependency injection
    """
    config = providers.Configuration()

    image_service = providers.Singleton(ImageService, config=config)

    object_service = providers.Singleton(ObjectService, config=config, image_service=image_service)

    pose_map_service = providers.Singleton(PoseMapService, config=config,
                                           object_service=object_service, image_service=image_service)

    contour_service = providers.Singleton(ContourService, config=config,
                                          image_service=image_service, object_service=object_service,
                                          pose_map_service=pose_map_service)

    tracker = providers.Singleton(Tracker, config=config, object_service=object_service,
                                  contour_service=contour_service)
