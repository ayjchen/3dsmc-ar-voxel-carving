import cv2
import cv2.aruco as aruco
import os
import numpy as np

def detect_aruco_markers(input_path, output_path):
    #Detect ArUco markers in an image and draw them on the output.
    # Read the input image
    input_image = cv2.imread(input_path)

    if input_image is None:
        print(f"Error: Unable to load image {input_path}")
        return

    # Detector parameters and dictionary
    detector_params = aruco.DetectorParameters()
    dictionary = aruco.getPredefinedDictionary(aruco.DICT_5X5_50)
    detector = aruco.ArucoDetector(dictionary, detector_params)

    # Detect ArUco markers
    marker_corners, marker_ids, _ = detector.detectMarkers(input_image)

    if marker_ids is not None and len(marker_ids) > 0:
        print(f"Detected {len(marker_ids)} ArUco markers in image '{os.path.basename(input_path)}': IDs = {marker_ids.flatten().tolist()}")

        # Draw detected markers on the image
        output_image = input_image.copy()
        aruco.drawDetectedMarkers(output_image, marker_corners, marker_ids, borderColor=(0, 255, 0))

        # Save the output image
        cv2.imwrite(output_path, output_image)
        print(f"Processed and saved: {output_path}")
    else:
        print(f"No ArUco markers detected in image '{os.path.basename(input_path)}'")

def process_dataset(input_dir, output_dir):
    """Process all images in a dataset to detect ArUco markers."""
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    for filename in os.listdir(input_dir):
        if filename.lower().endswith((".jpg", ".png", ".jpeg")):
            input_path = os.path.join(input_dir, filename)
            output_path = os.path.join(output_dir, filename)
            detect_aruco_markers(input_path, output_path)

if __name__ == "__main__":
    input_directory = "./data_raw_test"
    output_directory = "./data_arucodetector_test"
    process_dataset(input_directory, output_directory)
