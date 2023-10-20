# INT13146 Image Processing (Xu ly Anh)
# Nguyen Phan Nhut Truong N20DCCN082
import cv2
import numpy as np
import matplotlib.pyplot as plt

# Tải hình ảnh thang độ xám
img = cv2.imread("lady.bin", cv2.IMREAD_GRAYSCALE)

# Vẽ biểu đồ cho hình ảnh
plt.hist(img.ravel(), bins=256, range=(0, 256), color='gray')
plt.xlabel('Pixel Intensity')
plt.ylabel('Frequency')
plt.title('Histogram of "lady.bin"')
plt.show()

# Thực hiện kéo dài độ tương phản toàn diện
img_stretched = cv2.normalize(img, None, 0, 255, cv2.NORM_MINMAX)

# Vẽ biểu đồ cho kết quả
plt.hist(img_stretched.ravel(), bins=256, range=(0, 256), color='gray')
plt.xlabel('Pixel Intensity')
plt.ylabel('Frequency')
plt.title('Histogram of Contrast-Stretched "lady.bin"')
plt.show()


