from service.service_interface import IService


class ContourService(IService):
    image_path = None
    model_name = None
    image_service = None
    object_service = None

    def __init__(self, config, image_service, object_service):
        super().__init__(config)
        self.image_service = image_service
        self.object_service = object_service

    def simplify_contours(self):
        tracked_object = self.object_service.get_object()
        # tracked_object.set_contours() TODO implement this
        pass
