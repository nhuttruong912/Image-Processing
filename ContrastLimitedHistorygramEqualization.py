# INT13146 Image Processing (Xu ly Anh)
# Nguyen Phan Nhut Truong N20DCCN082


import cv2
import numpy as np
import matplotlib.pyplot as plt

image_path = 'brain.jpg'
equalized_path = 'equalized_image.jpg'

image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
equalized_image = cv2.equalizeHist(image)

fig, axes = plt.subplots(1, 4, figsize=(16, 8))
figManager = plt.get_current_fig_manager()
axes[0].imshow(image, cmap='gray')
axes[0].set_title('Original Image')

axes[1].imshow(equalized_image, cmap='gray')
axes[1].set_title('Equalized Image')

axes[2].hist(image.ravel(), 256, [0, 256])
axes[2].set_title('Histogram (Original)')
axes[2].set_xlim(0, 255)

axes[3].hist(equalized_image.ravel(), 256, [0, 256])
axes[3].set_title('Histogram (Equalized)')
axes[3].set_xlim(0, 255)

plt.tight_layout()
plt.bar(np.arange(256), 0)
plt.show()