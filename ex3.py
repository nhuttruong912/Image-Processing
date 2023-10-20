# INT13146 Image Processing (Xu ly Anh)
# Nguyen Phan Nhut Truong N20DCCN082
import cv2
import numpy as np

# Tải ảnh nhị phân
img = cv2.imread("actontBin.bin", cv2.IMREAD_GRAYSCALE)

# Xác định mẫu
template = np.array([
    [0, 255, 0],
    [0, 255, 0],
    [0, 255, 0]
], dtype=np.uint8)

# Áp dụng thuật toán So khớp mẫu nhị phân
J1 = cv2.matchTemplate(img, template, cv2.TM_CCOEFF_NORMED)

# Xây dựng hình ảnh đầu ra J1 trong đó mỗi pixel bằng độ đo khớp M2
J1[J1 < 0.5] = 0

# Ngưỡng J1 để đạt J2
threshold = np.max(J1) * 0.9
J2 = np.where(J1 > threshold, 255, 0)

# Lưu J2 dưới dạng ảnh nhị phân
cv2.imwrite("J2.png", J2)
