import cv2
import numpy as np
import os

def remove_foreground(image_path, output_path):
    # Read the image
    img = cv2.imread(image_path)
    if img is None:
        print(f"Failed to load image {image_path}")
        return

    # Create a mask
    mask = np.zeros(img.shape[:2], np.uint8)

    # Define the background and foreground models
    bgd_model = np.zeros((1, 65), np.float64)
    fgd_model = np.zeros((1, 65), np.float64)

    # Define the rectangle for the GrabCut algorithm
    rect = (50, 50, img.shape[1] - 50, img.shape[0] - 50)

    # Apply the GrabCut algorithm
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
    input_directory = "./data_raw"
    output_directory = "./data_preprocessed"
    process_dataset(input_directory, output_directory)