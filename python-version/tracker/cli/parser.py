from argparse import ArgumentParser

parser = ArgumentParser()

parser.add_argument("image_folder_path", help="The path to the folder of images you want to analyze")

args = parser.parse_args()


def add_parser_get_args():
    return args


def get_image_path():
    return args.image_folder_path
