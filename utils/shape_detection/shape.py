import os

import numpy as np
from PIL import Image


def shape_detection_by_dtw(detection_for: str):
    dynamic_time_warping = dynamic_time_warping_method()

    # Load and convert reference image to a sequence
    image_to_sequence = image_sequence_conversion()
    reference_image = ''
    other_images = []
    directory_name = ''
    # Example reference image
    if detection_for == 'pothole':
        directory_name = '../../pothole_data_set'
        reference_image = str(directory_name) + '/ref_img/refImg.jpg'
        for filename in os.listdir(directory_name):
            # Check if the file is a .jpg image
            if filename.endswith(".jpg"):
                # Append the file with directory name to the list
                other_images.append(os.path.join(directory_name, filename))

    reference_seq = image_to_sequence(reference_image)

    # List of other images to compare

    # Compare the reference image with each other image and print results
    for idx, image_path in enumerate(other_images, start=1):
        other_seq = image_to_sequence(image_path)
        dtw_distance = dynamic_time_warping(reference_seq, other_seq)

        # Calculate the threshold as 30% of the average sequence length
        threshold = 14
        # threshold = 0.3 * (len(reference_seq) + len(other_seq)) / 2

        # Compare the DTW distance with the threshold
        if dtw_distance < threshold:
            print(f"Image {idx}: Matched (DTW Distance: {dtw_distance:.4f})")
        else:
            print(f"Image {idx}: Not Matched (DTW Distance: {dtw_distance:.4f})")


def image_sequence_conversion():
    def image_to_sequence(image_path):
        image = Image.open(image_path).convert('L')
        image_array = np.array(image)
        normalized_image_array = image_array / 255.0
        return normalized_image_array

    return image_to_sequence


def dynamic_time_warping_method():
    def dynamic_time_warping(seq1, seq2):
        n = len(seq1)
        m = len(seq2)
        dtw_matrix = np.zeros((n + 1, m + 1))

        for i in range(1, n + 1):
            for j in range(1, m + 1):
                cost = abs(seq1[i - 1][j - 1] - seq2[i - 1][j - 1])  # Compare individual elements in lists
                dtw_matrix[i, j] = cost + min(dtw_matrix[i - 1, j], dtw_matrix[i, j - 1], dtw_matrix[i - 1, j - 1])

        return dtw_matrix[n, m]

    return dynamic_time_warping


def shape_detection_by_dtw_algorithm(detection_for: str):
    shape_detection_by_dtw(detection_for)
