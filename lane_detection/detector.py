import sys
import os
import cv2
import numpy as np
import logging

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from lane_detection.utils import check_image_validity

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

class LaneDetector:
    def __init__(self):
        pass

    def canny(self, image):
        """
        Applies Canny edge detection to an image.

        Parameters:
            image (numpy.ndarray): Input image in RGB format.

        Returns:
            numpy.ndarray: Canny edge-detected image.
        """
        if not check_image_validity(image):
            logging.error("Invalid image input for Canny edge detection.")
            raise ValueError("Invalid image input for Canny edge detection.")

        gray_image = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
        blur_image = cv2.GaussianBlur(gray_image, (5, 5), 0)
        canny_image = cv2.Canny(blur_image, 50, 150)
        return canny_image

    def region_of_interest(self, image):
        """
        Applies an image mask to keep only the region of interest.

        Parameters:
            image (numpy.ndarray): Input image.

        Returns:
            numpy.ndarray: Masked image with region of interest.
        """
        if image is None :
            raise ValueError("image can not be none")
        
        height, width = image.shape[:2]
        polygon = np.array([
            [
                (int(width * 0.30), int(height * 0.75)),
                (int(width * 0.75), int(height * 0.75)),
                (int(width * 0.60), int(height * 0.6)),
                (int(width * 0.45), int(height * 0.6))
            ]
        ])

        mask = np.zeros_like(image)
        cv2.fillPoly(mask, polygon, 255)
        masked_image = cv2.bitwise_and(image, mask)
        return masked_image

    def avg_slope_intercept(self, image, lines):
        """
        Calculates the average slope and intercept for left and right lane lines.

        Parameters:
            image (numpy.ndarray): Input image.
            lines (numpy.ndarray): Detected lines from Hough transform.

        Returns:
            numpy.ndarray: Coordinates of the averaged left and right lane lines.
        """
        left_fit = []
        right_fit = []

        if lines is None or len(lines) == 0:
            logging.warning("No lines detected for slope calculation.")
            return np.array([])

        for line in lines:
            x1, y1, x2, y2 = line.reshape(4)
            parameters = np.polyfit((x1, x2), (y1, y2), 1)
            slope = parameters[0]
            intercept = parameters[1]

            if slope < 0:
                left_fit.append((slope, intercept))
            else:
                right_fit.append((slope, intercept))

        if len(left_fit) == 0 or len(right_fit) == 0:
            logging.warning("Insufficient data for line fitting.")
            return np.array([])

        left_fit_avg = np.average(left_fit, axis=0)
        right_fit_avg = np.average(right_fit, axis=0)

        left_line = self.make_coordinates(image, left_fit_avg)
        right_line = self.make_coordinates(image, right_fit_avg)

        return np.array([left_line, right_line])

    def make_coordinates(self, image, line_parameters):
        """
        Generates the coordinates for the line given its slope and intercept.

        Parameters:
            image (numpy.ndarray): Input image.
            line_parameters (tuple): Slope and intercept of the line.

        Returns:
            numpy.ndarray: Coordinates of the line in the image.
        """
        slope, intercept = line_parameters

        if slope == 0:
            logging.warning("Slope is zero, invalid line parameters.")
            return np.array([])

        y1 = image.shape[0]
        y2 = int(y1 * (3 / 5))

        x1 = int((y1 - intercept) / slope)
        x2 = int((y2 - intercept) / slope)

        x1 = max(0, min(x1, image.shape[1] - 1))
        x2 = max(0, min(x2, image.shape[1] - 1))

        return np.array([x1, y1, x2, y2])

    def fill_lane(self, image, lines):
        """
        Fills the detected lane area with a color.

        Parameters:
            image (numpy.ndarray): Input image.
            lines (numpy.ndarray): Coordinates of the lane lines.

        Returns:
            numpy.ndarray: Image with filled lane area.
        """
        lane_fill_image = np.zeros_like(image)
        if lines is not None and len(lines) == 2:
            left_line = lines[0]
            right_line = lines[1]
            left_points = [(left_line[0], left_line[1]), (left_line[2], left_line[3])]
            right_points = [(right_line[2], right_line[3]), (right_line[0], right_line[1])]
            lane_points = np.array([left_points + right_points])
            cv2.fillPoly(lane_fill_image, lane_points, (255, 0, 0))

        return lane_fill_image

    def process_frame(self, frame):
        """
        Processes a single frame of video to detect and fill lanes.

        Parameters:
            frame (numpy.ndarray): Input video frame.

        Returns:
            numpy.ndarray: Processed frame with detected lanes.
        """
        if frame is None:
            logging.error("Invalid frame input for processing.")
            raise ValueError("Invalid frame input for processing.")

        canny_image = self.canny(frame)
        cropped_image = self.region_of_interest(canny_image)
        lines = cv2.HoughLinesP(cropped_image, 10, np.pi / 180, 100, minLineLength=50, maxLineGap=5)
        avg_lines = self.avg_slope_intercept(frame, lines)
        filled_lane_image = self.fill_lane(frame, avg_lines)
        combo_image = cv2.addWeighted(frame, 0.8, filled_lane_image, 1, 1)
        return combo_image
