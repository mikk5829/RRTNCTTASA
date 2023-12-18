import cv2 as cv
import numpy as np

class RobustMatcher:
    def __init__(self):
        self.detector_ = cv.ORB.create()
        self.extractor_ = cv.ORB.create()
        self.matcher_ = cv.BFMatcher(cv.NORM_HAMMING, crossCheck=True)
        self.ratio_ = 0.8

    def compute_keypoints(self, image):
        keypoints = self.detector_.detect(image, None)
        return keypoints

    def compute_descriptors(self, image, keypoints):
        keypoints, descriptors = self.extractor_.compute(image, keypoints)
        return keypoints, descriptors

    def ratio_test(self, matches):
        good = []
        for m, n in matches:
            if m.distance < self.ratio_ * n.distance:
                good.append(m)
        return good

    def symmetry_test(self, matches1, matches2):
        sym_matches = []
        for match1 in matches1:
            for match2 in matches2:
                if match1.queryIdx == match2.trainIdx and match2.queryIdx == match1.trainIdx:
                    sym_matches.append(match1)
                    break
        return sym_matches

    def robust_match(self, frame, descriptors_model, keypoints_model):
        keypoints_frame = self.compute_keypoints(frame)
        keypoints_frame, descriptors_frame = self.compute_descriptors(frame, keypoints_frame)

        matches12 = self.matcher_.knnMatch(descriptors_frame, descriptors_model, k=2)
        matches21 = self.matcher_.knnMatch(descriptors_model, descriptors_frame, k=2)

        matches12 = self.ratio_test(matches12)
        matches21 = self.ratio_test(matches21)

        good_matches = self.symmetry_test(matches12, matches21)

        return good_matches, keypoints_frame

    def fast_robust_match(self, frame, descriptors_model, keypoints_model):
        keypoints_frame = self.compute_keypoints(frame)
        keypoints_frame, descriptors_frame = self.compute_descriptors(frame, keypoints_frame)

        matches = self.matcher_.knnMatch(descriptors_frame, descriptors_model, k=2)
        good_matches = self.ratio_test(matches)

        return good_matches, keypoints_frame