class IService:
    """
    This is the interface for all services
    """
    image_path = None
    model_name = None
    path_to_model_images = None

    def __init__(self, config):
        self.image_path = config["image_path"]
        self.model_name = config["model_name"]
        self.path_to_model_images = config["path_to_model_images"]
