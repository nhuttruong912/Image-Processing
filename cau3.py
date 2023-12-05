import numpy as np
import cv2

def binary_template_matching(input_image, template):
    match_measure = cv2.matchTemplate(input_image, template, cv2.TM_SQDIFF_NORMED)
    return match_measure

def threshold_image(image, threshold):
    _, binary_image = cv2.threshold(image, threshold, 1, cv2.THRESH_BINARY)
    return binary_image.astype(np.uint8) * 255

# Load the binary image
input_image_path = "actontBin.bin"
input_image = np.fromfile(input_image_path, dtype=np.uint8)
input_image = input_image.reshape((256, 256))

# Define your template based on the shape you're looking for
template = np.array([[0, 1, 0],
                     [1, 1, 1],
                     [0, 1, 0]], dtype=np.uint8)

# Apply Binary Template Matching (Match Measure M2)
match_measure = binary_template_matching(input_image, template)

# Construct Output Image J1
output_image_J1 = match_measure

# Threshold Image J1
threshold_value = 0.9  # Adjust this threshold value as needed
binary_image_J2 = threshold_image(output_image_J1, threshold_value)

# Display the binary image J2
cv2.imshow("Binary Image J2", binary_image_J2)
cv2.waitKey(0)
cv2.destroyAllWindows()
