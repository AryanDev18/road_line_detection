import numpy as np

def check_image_validity(image):
    """
    Checks if the image is valid for processing.

    Parameters:
        image (numpy.ndarray): Input image.

    Returns:
        bool: True if image is valid, False otherwise.
    """
    return image is not None and isinstance(image, np.ndarray) and len(image.shape) == 3
