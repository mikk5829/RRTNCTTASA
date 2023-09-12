from argparse import ArgumentParser
from pathvalidate.argparse import validate_filename_arg

parser = ArgumentParser()

parser.add_argument("image_path", help="The path to the image you want to analyze", type=validate_filename_arg)

args = parser.parse_args()


def add_parser_get_args():
    return args
