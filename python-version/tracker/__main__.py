from cli.parser import Parser
from image.pose_map_service import PoseMapService
from tracker import Tracker
from dependency_injector.wiring import Provide, inject
from containers import Container
import seaborn as sns


@inject
def use_case(
        tracker: Tracker = Provide[Container.tracker],
        pose_map_service: PoseMapService = Provide[Container.pose_map_service],
        config: dict = Provide[Container.config]
) -> None:
    if config["path_to_model_images"] is not None and config["track"] is True:
        tracker.estimate_poses()
    if config["path_to_model_images"] is not None and config["track"] is False:
        pose_map_service.set_new_pose_map()
    if config["image_path"] is not None:
        tracker.estimate_pose()
    pass


def main():
    sns.set()
    sns.set_context("paper", font_scale=1.2)
    # seaborn set retina display

    parser = Parser()
    options = parser.add_parser_get_args()

    container = Container()

    # Prepend all keys in options with "__"
    container.config.from_dict(options.__dict__)

    container.init_resources()
    container.wire(modules=[__name__])

    use_case()


if __name__ == "__main__":
    main()
