# INT13146 Image Processing (Xu ly Anh)
# Nguyen Phan Nhut Truong N20DCCN082
import cv2
import numpy as np
import matplotlib.pyplot as plt

# Load the grayscale image
img = cv2.imread("johnny.bin", cv2.IMREAD_GRAYSCALE)

# Plot a histogram for the image
plt.hist(img.ravel(), bins=256, range=(0, 256), color='gray')
plt.xlabel('Pixel Intensity')
plt.ylabel('Frequency')
plt.title('Histogram of Original Image')
plt.show()

# Perform histogram equalization
img_eq = cv2.equalizeHist(img)

# Plot a histogram for the result
plt.hist(img_eq.ravel(), bins=256, range=(0, 256), color='gray')
plt.xlabel('Pixel Intensity')
plt.ylabel('Frequency')
plt.title('Histogram of Equalized Image')
plt.show()

# Display the equalized image
cv2.imshow("Equalized Image", img_eq)
cv2.waitKey(0)
cv2.destroyAllWindows()
