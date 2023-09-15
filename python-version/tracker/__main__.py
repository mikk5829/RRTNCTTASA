from tracker.image.photo import get_images_from_directory
from tracker.cli.parser import add_parser_get_args


# Defining main function
def main():
    args = add_parser_get_args()
    images = get_images_from_directory(args.image_folder_path)
    # print keys
    print("The following images were found in the folder:")
    for key in images.keys():
        print(key)


if __name__ == "__main__":
    main()