import numpy as np
import matplotlib.pyplot as plt

def plot_histogram(image, title):
    hist, bins = np.histogram(image.flatten(), bins=256, range=(0,256))
    plt.figure(figsize=(10, 5))
    plt.subplot(121)
    plt.plot(hist, color='black')
    plt.title(f'{title} Histogram')

def histogram_equalization(image):
    hist, bins = np.histogram(image.flatten(), bins=256, range=(0,256))
    cdf = hist.cumsum()
    cdf_normalized = cdf * hist.max() / cdf.max()
    equalized_image = np.interp(image.flatten(), bins[:-1], cdf_normalized)
    return equalized_image.reshape(image.shape).astype(np.uint8)

# Load the original image
image_path = "johnny.bin"
original_image = np.fromfile(image_path, dtype=np.uint8)
original_image = original_image.reshape((256, 256))

# Plot the histogram of the original image
plot_histogram(original_image, "Original Image")

# Perform histogram equalization
equalized_image = histogram_equalization(original_image)

# Plot the histogram of the equalized image
plot_histogram(equalized_image, "Equalized Image")

# Show the original and equalized images
plt.subplot(122)
plt.imshow(equalized_image, cmap='gray', vmin=0, vmax=255)
plt.title('Equalized Image')
plt.show()
