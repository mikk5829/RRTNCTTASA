from service.service_interface import IService


class ContourService(IService):
    image_path = None
    model_name = None
    image_service = None

    def __init__(self, image_path, model_name, image_service):
        super().__init__(image_path, model_name)
        self.image_service = image_service

    def get_contours_from_image(self):
        image = self.image_service.get_image()
        print(image)
