import numpy as np

def simple_thresholding(image, threshold):
    binary_image = np.where(image >= threshold, 255, 0)
    return binary_image.astype(np.uint8)

# Load the grayscale image
image_path = "mammogram.bin"
image = np.fromfile(image_path, dtype=np.uint8)
image = image.reshape((256, 256))

# Define the threshold value (you may need to experiment to find a suitable value)
threshold = 120

# Perform simple thresholding
binary_image = simple_thresholding(image, threshold)

# Save the binary image (optional)
binary_image_path = "binary_mammogram.bin"
binary_image.tofile(binary_image_path)

# Display the binary image
# (You may use your preferred method for displaying images)
# For example, using Matplotlib:
import matplotlib.pyplot as plt
plt.imshow(binary_image, cmap='gray', vmin=0, vmax=255)
plt.show()