# INT13146 Image Processing (Xu ly Anh)
# Nguyen Phan Nhut Truong N20DCCN082

import cv2

# Tải hình ảnh thang độ xám
img = cv2.imread("Mammogram.bin", cv2.IMREAD_GRAYSCALE)

# Tính giá trị ngưỡng
_, thresh = cv2.threshold(img, 0, 255, cv2.THRESH_BINARY+cv2.THRESH_OTSU)

# Lưu ảnh nhị phân
cv2.imwrite("binary_image.png", thresh)

thresh[thresh == 255] = 0xff
thresh[thresh == 0] = 0x00
cv2.imwrite("binary_image.png", thresh)

