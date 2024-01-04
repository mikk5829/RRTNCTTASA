class IService:
    """
    This is the interface for all services
    """
    image_path: str = None
    model_name: str = None
    path_to_model_images: str = None
    verbose: bool = False
    track: bool = False

    def __init__(self, config):
        self.image_path = config["image_path"]
        self.model_name = config["model_name"]
        self.path_to_model_images = config["path_to_model_images"]
        self.verbose = config["verbose"]
        self.track = config["track"]
