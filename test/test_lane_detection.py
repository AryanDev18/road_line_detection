import sys
import os
import pytest
import numpy as np

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from lane_detection.detector import LaneDetector

def test_canny_invalid_input():
    detector = LaneDetector()
    with pytest.raises(ValueError):
        detector.canny(None)

def test_region_of_interest_invalid_input():
    detector = LaneDetector()
    with pytest.raises(ValueError):
        detector.region_of_interest(None)

def test_make_coordinates_zero_slope():
    detector = LaneDetector()
    image = np.zeros((480, 640, 3), dtype=np.uint8)
    coordinates = detector.make_coordinates(image, (0, 10))
    assert len(coordinates) == 0

def test_avg_slope_intercept_no_lines():
    detector = LaneDetector()
    image = np.zeros((480, 640, 3), dtype=np.uint8)
    lines = np.array([])
    avg_lines = detector.avg_slope_intercept(image, lines)
    assert len(avg_lines) == 0

def test_fill_lane_empty_lines():
    detector = LaneDetector()
    image = np.zeros((480, 640, 3), dtype=np.uint8)
    filled_image = detector.fill_lane(image, np.array([]))
    assert np.array_equal(filled_image, image)

def test_check_image_validity():
    from lane_detection.utils import check_image_validity

    valid_image = np.zeros((480, 640, 3), dtype=np.uint8)
    invalid_image = None
    invalid_type_image = "not_an_image"

    assert check_image_validity(valid_image) is True
    assert check_image_validity(invalid_image) is False
    assert check_image_validity(invalid_type_image) is False
