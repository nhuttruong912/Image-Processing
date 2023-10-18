# INT13146 Image Processing (Xu ly Anh)
# Nguyen Phan Nhut Truong N20DCCN082
import cv2
import numpy as np
import matplotlib.pyplot as plt

# Step 1: Tải ảnh gốc
image_path = 'parrot.jpg'  # Replace with the path to your Parrot image
image = cv2.imread(image_path)

# Step 2: Chuyển đổi hình ảnh sang thang độ xám
gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

# Step 3: Áp dụng cân bằng biểu đồ toàn cầu
global_equalized = cv2.equalizeHist(gray_image)

# Step 4: Tạo object CLAHE cho ô 8x8
clahe_8x8 = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
clahe_8x8_equalized = clahe_8x8.apply(gray_image)

# Step 5: Tạo object CLAHE cho ô 16x16
clahe_16x16 = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(16, 16))
clahe_16x16_equalized = clahe_16x16.apply(gray_image)

# Step 6: Vẽ hình ảnh gốc và cân bằng
plt.figure(figsize=(12, 12))

plt.subplot(2, 2, 1)
plt.title("Original Image")
plt.imshow(gray_image, cmap='gray')
plt.axis('off')

plt.subplot(2, 2, 2)
plt.title("Global Histogram Equalization")
plt.imshow(global_equalized, cmap='gray')
plt.axis('off')

plt.subplot(2, 2, 3)
plt.title("Adaptive Histogram Equalization (8x8 tiles)")
plt.imshow(clahe_8x8_equalized, cmap='gray')
plt.axis('off')

plt.subplot(2, 2, 4)
plt.title("Adaptive Histogram Equalization (16x16 tiles)")
plt.imshow(clahe_16x16_equalized, cmap='gray')
plt.axis('off')

plt.tight_layout()
plt.show()

