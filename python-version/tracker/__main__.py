from cli.parser import Parser
from image.pose_map_service import PoseMapService
from tracker import Tracker
from dependency_injector.wiring import Provide, inject
from containers import Container


@inject
def use_case(
        tracker: Tracker = Provide[Container.tracker]
) -> None:
    tracker.estimate_pose()
    pass


def main():
    parser = Parser()
    options = parser.add_parser_get_args()

    container = Container()

    # set the options in the container config
    for key, value in options.__dict__.items():
        if value is not None:
            container.config[key].from_value(value)

    container.init_resources()
    container.wire(modules=[__name__])

    use_case()


if __name__ == "__main__":
    main()
