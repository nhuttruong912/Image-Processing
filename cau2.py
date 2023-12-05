import numpy as np
import matplotlib.pyplot as plt

# Load the original image
image_path = "lady.bin"
image = np.fromfile(image_path, dtype=np.uint8)
image = image.reshape((256, 256))

# Plot histogram for the original image
hist_original, bins_original = np.histogram(image, bins=256, range=(0, 256))
plt.figure(figsize=(10, 5))
plt.subplot(121)
plt.plot(hist_original, color='black')
plt.title('Original Image Histogram')

# Define input and output intensity bounds
min_input, max_input = np.min(image), np.max(image)
min_output, max_output = 0, 255

# Perform contrast stretch
contrast_stretched = ((image - min_input) / (max_input - min_input) * (max_output - min_output)) + min_output

# Plot histogram for the contrast-stretched image
hist_stretched, bins_stretched = np.histogram(contrast_stretched, bins=256, range=(0, 256))
plt.subplot(122)
plt.plot(hist_stretched, color='black')
plt.title('Contrast Stretched Histogram')

# Display the images and histograms
plt.show()