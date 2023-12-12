import os
import cv2 as cv
from tqdm import tqdm


def get_files_from_directory(folder_path) -> list[str]:
    """
    This function is used to get all the files from a folder
    :param folder_path: the path to the folder with the files
    :return: list with files
    """
    files = list()
    # find files in folder
    for filename in sorted(os.listdir(folder_path)):
        # append files to list
        files.append(filename)

    # return list with files
    return files


for file in tqdm(get_files_from_directory("../blender_images")):
    # read and save red, green and blue channels
    img = cv.imread("../blender_images/" + file)
    if img is None:
        continue
    blue_channel, green_channel, red_channel = cv.split(img)

    # combine channels r + g, r + b and g + b
    cv.imwrite("/Users/mikkel/repos/RRTNCTTASA/blender_images/rgb/rg_" + file, cv.add(red_channel, green_channel))
    cv.imwrite("/Users/mikkel/repos/RRTNCTTASA/blender_images/rgb/rb_" + file, cv.add(red_channel, blue_channel))
    cv.imwrite("/Users/mikkel/repos/RRTNCTTASA/blender_images/rgb/gb_" + file, cv.add(green_channel, blue_channel))

    # save channels
    # cv.imwrite("/Users/mikkel/repos/RRTNCTTASA/blender_images/rgb/r_" + file, red_channel)
    # cv.imwrite("/Users/mikkel/repos/RRTNCTTASA/blender_images/rgb/g_" + file, green_channel)
    # cv.imwrite("/Users/mikkel/repos/RRTNCTTASA/blender_images/rgb/b_" + file, blue_channel)
