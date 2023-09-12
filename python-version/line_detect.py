"""
@file hough_lines.py
@brief This program demonstrates line finding with the Hough transform
"""
import sys
import math
import cv2 as cv
import numpy as np
import matplotlib.pyplot as plt


def main(argv):

    ## [load]
    default_file = 'Line4Detection.jpg'
    # default_file = 'Proba-3_formation-flying.jpg'
    filename = argv[0] if len(argv) > 0 else default_file

    # Loads an image
    src = cv.imread(cv.samples.findFile(filename), cv.IMREAD_GRAYSCALE)

    src = cv.Canny(src, 80, 150)

    lsd1 = cv.LineSegmentDetector('Refine','Standard')
    lsd2 = cv.LineSegmentDetector('Refine','None')

    #lines1 = lsd1.detect(src)
    #lines2 = lsd2.detect(src)

    #drawnLines1 = lsd1.drawSegments(src, lines1)
    #imshow(drawnLines1), title('Standard refinement')

    #drawnLines2 = lsd2.drawSegments(src, lines2)
    #imshow(drawnLines2), title('No refinement')

    # [h,w,~] = size(img);
    #[comparison,mis_count] = lsd1.compareSegments([w,h], lines1, lines2);
    #imshow(comparison), title(sprintf('Mismatch = %d', mis_count))


if __name__ == "__main__":
    main(sys.argv[1:])