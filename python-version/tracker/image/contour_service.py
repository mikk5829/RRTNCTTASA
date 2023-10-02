from service.service_interface import IService


class ContourService(IService):
    image_path = None
    model_name = None

    def __init__(self, image_path, model_name):
        super().__init__(image_path, model_name)
