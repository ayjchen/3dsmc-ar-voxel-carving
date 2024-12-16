import cv2
import cv2.aruco as aruco
import numpy as np
import os

def create_grabcut_bbox(marker_corners, image_shape):
    #Create a bounding box for GrabCut based on detected ArUco marker corners.
    all_corners = np.concatenate(marker_corners, axis=1)
    x_min = max(0, int(np.min(all_corners[:, :, 0])))
    x_max = min(image_shape[1], int(np.max(all_corners[:, :, 0])))
    y_min = max(0, int(np.min(all_corners[:, :, 1])))
    y_max = min(image_shape[0], int(np.max(all_corners[:, :, 1])))
    return (x_min, y_min, x_max, y_max)

def remove_foreground(image_path, output_path):
    # Read the image
    img = cv2.imread(image_path)
    if img is None:
        print(f"Failed to load image {image_path}")
        return
    
    # Detector parameters and dictionary
    detector_params = aruco.DetectorParameters()
    dictionary = aruco.getPredefinedDictionary(aruco.DICT_5X5_50)
    detector = aruco.ArucoDetector(dictionary, detector_params)

    # Detect ArUco markers
    marker_corners, marker_ids, _ = detector.detectMarkers(img)

    if marker_ids is None or len(marker_ids) == 0:
        print(f"No ArUco markers detected in image '{os.path.basename(image_path)}'")
        return
    
    print(f"Detected {len(marker_ids)} ArUco markers in image '{os.path.basename(image_path)}': IDs = {marker_ids.flatten().tolist()}")

    # Create a bounding box from the detected markers
    bbox = create_grabcut_bbox(marker_corners, img.shape)
    print(f"Bounding box for GrabCut: {bbox}")

    # Create a mask
    mask = np.zeros(img.shape[:2], np.uint8)

    # Define the background and foreground models
    bgd_model = np.zeros((1, 65), np.float64)
    fgd_model = np.zeros((1, 65), np.float64)

    # Define the rectangle for the GrabCut algorithm
    rect = (50, 50, img.shape[1] - 50, img.shape[0] - 50)

    # Apply GrabCut algorithm
    rect = (bbox[0], bbox[1], bbox[2] - bbox[0], bbox[3] - bbox[1])
    cv2.grabCut(img, mask, rect, bgd_model, fgd_model, 5, cv2.GC_INIT_WITH_RECT)

    # Modify the mask
    mask2 = np.where((mask == 2) | (mask == 0), 0, 1).astype('uint8')

    # Apply the mask to the image
    img = img * mask2[:, :, np.newaxis]

    # Save the result
    cv2.imwrite(output_path, img)

def process_dataset(input_dir, output_dir):
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    for filename in os.listdir(input_dir):
        if filename.lower().endswith((".jpg", ".png", ".jpeg")):
            input_path = os.path.join(input_dir, filename)
            output_path = os.path.join(output_dir, filename)
            remove_foreground(input_path, output_path)

if __name__ == "__main__":
    input_directory = "./data_raw_test"
    output_directory = "./data_preprocessed_test"
    process_dataset(input_directory, output_directory)