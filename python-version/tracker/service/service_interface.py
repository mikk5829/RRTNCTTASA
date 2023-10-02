class IService:
    """
    This is the interface for all services
    """
    image_path = None
    model_name = None

    def __init__(self, image_path, model_name):
        self.image_path = image_path
        self.model_name = model_name
