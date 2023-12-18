from PIL import Image, PngImagePlugin
import numpy as np
import struct
import matplotlib.pyplot as plt
import glob

def read_header(fid):
    """
    Reads the header information from the given file object.

    Parameters:
    fid (file object): The file object to read from.

    Returns:
    dict: A dictionary containing the header information.
    """
    header = {}

    temp = struct.unpack('HH', fid.read(4))
    header['DAC_OFFFSET'] = temp[0]
    header['DAC_gain'] = temp[1]

    temp = struct.unpack('BBBBBB', fid.read(6))
    header['INTEGRATION_TIME'] = temp[0]
    header['COMPRESSION'] = temp[1]
    header['ROI'] = temp[2]
    header['JPEG_QUALITY'] = temp[3]
    header['COMPRESSION_THRESHOLD'] = temp[4]
    header['INFO'] = temp[5]

    temp = struct.unpack('HH', fid.read(4))
    header['VALID'] = temp[0]
    header['STATUS'] = temp[1]

    temp = struct.unpack('II', fid.read(8))
    header['CODE_START'] = temp[0]
    header['CODE_END'] = temp[1]

    temp = struct.unpack('H', fid.read(2))
    header['SUB_TIMESTAMP'] = temp[0]

    temp = struct.unpack('I', fid.read(4))
    header['TIMESTAMP'] = temp[0]

    temp = struct.unpack('HHH', fid.read(6))
    header['H'] = temp[0]
    header['W'] = temp[1]
    header['IMOD'] = temp[2]

    return header


def unc_to_png(unc_file_path, tif_file_path):
    """
    Convert an UNC file to a PNG image.

    Args:
        unc_file_path (str): The path to the UNC file.
        tif_file_path (str): The path to save the PNG image.

    Returns:
        None
    """
    with open(unc_file_path, 'rb') as f:
        # Skip the header
        header = read_header(f)

        # Reshape the image data into a 2D array
        n = header['H'] * (header['IMOD'] + 1) * header['W']
        temp = np.fromfile(f, dtype=np.uint8, count=n)
        image = np.reshape(temp, (header['H'] * (header['IMOD'] + 1), header['W']))

        # Create a PIL image from the numpy array
        im = Image.fromarray(image)

        # Save the image as a .png file
        im.save(tif_file_path, format='png')


# Get all .unc files in the directory
files = glob.glob(
    'test_images/theta15deg_8steps_45degs/**/*.unc',
    recursive=True)
# for each file, run unc_to_png
np.vectorize(unc_to_png)(files, [f.replace('.unc', '.png') for f in files])
