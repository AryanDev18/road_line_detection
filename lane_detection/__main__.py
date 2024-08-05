import sys
import os
import cv2
import logging

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from lane_detection.detector import LaneDetector

def main(video_path):
    """
    Main function to process video for lane detection.

    Parameters:
        video_path (str): Path to the input video file.
    """
    # Set up logging
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

    lane_detector = LaneDetector()

    if not os.path.exists(video_path):
        logging.error(f"The video file at '{video_path}' does not exist.")
        return

    cap = cv2.VideoCapture(video_path)

    if not cap.isOpened():
        logging.error("Could not open video file.")
        return

    while cap.isOpened():
        ret, frame = cap.read()

        if not ret:
            logging.info("End of video or error reading frame.")
            break

        try:
            processed_frame = lane_detector.process_frame(frame)
            cv2.imshow('Lane Detection', processed_frame)
        except Exception as e:
            logging.error(f"An error occurred during frame processing: {e}")

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    video_file_path = 'D:/dummy project/test03.mov'  # Update this path with your video file path
    main(video_file_path)
