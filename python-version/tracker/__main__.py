from tracker.cli.parser import add_parser_get_args


# Defining main function
def main():
    args = add_parser_get_args()
    print(f"File name: {args.image_path}")


if __name__ == "__main__":
    main()