from argparse import ArgumentParser, RawDescriptionHelpFormatter


class Parser:
    banner = """
        ██████╗ ██████╗ ████████╗███╗   ██╗ ██████╗████████╗████████╗ █████╗ ███████╗ █████╗ 
        ██╔══██╗██╔══██╗╚══██╔══╝████╗  ██║██╔════╝╚══██╔══╝╚══██╔══╝██╔══██╗██╔════╝██╔══██╗
        ██████╔╝██████╔╝   ██║   ██╔██╗ ██║██║        ██║      ██║   ███████║███████╗███████║
        ██╔══██╗██╔══██╗   ██║   ██║╚██╗██║██║        ██║      ██║   ██╔══██║╚════██║██╔══██║
        ██║  ██║██║  ██║   ██║   ██║ ╚████║╚██████╗   ██║      ██║   ██║  ██║███████║██║  ██║
        ╚═╝  ╚═╝╚═╝  ╚═╝   ╚═╝   ╚═╝  ╚═══╝ ╚═════╝   ╚═╝      ╚═╝   ╚═╝  ╚═╝╚══════╝╚═╝  ╚═╝
        Robust Real-Time Non-Cooperative Target Tracking Algorithm for Space Applications"""
    parser = ArgumentParser(
        formatter_class=RawDescriptionHelpFormatter,
        description=banner
    )

    parser.add_argument("--folder", help="The path to the folder of images you want to analyze")
    parser.add_argument("--image-path", help="The path to the image you want to analyze")
    parser.add_argument("--path-to-model-images", help="The path to the images you want to add to the pose "
                                                       "map")
    # adding required arguments
    parser.add_argument("model_name", help="The name for the model you want to use, if no image path is provided, "
                                           "then the output will be the last Pose recorded")

    args, unknown = parser.parse_known_args()

    def add_parser_get_args(self):
        return self.args
