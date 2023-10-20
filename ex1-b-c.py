# INT13146 Image Processing (Xu ly Anh)
# Nguyen Phan Nhut Truong N20DCCN082
import cv2
import numpy as np
# Tải hình ảnh thang độ xám
img = cv2.imread("Mammogram.bin", cv2.IMREAD_GRAYSCALE)

# Tính giá trị ngưỡng
_, thresh = cv2.threshold(img, 0, 255, cv2.THRESH_BINARY+cv2.THRESH_OTSU)

# Lưu ảnh nhị phân
cv2.imwrite("Mammogram_binary.png", thresh)

# Tải ảnh nhị phân
img = cv2.imread("Mammogram_binary.png", cv2.IMREAD_GRAYSCALE)

# Tìm đường viền trong ảnh nhị phân
contours, hierarchy = cv2.findContours(img, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

# Đường viền gần đúng bằng thuật toán Douglas-Peucker
epsilon = 0.01 * cv2.arcLength(contours[0], True)
approx = cv2.approxPolyDP(contours[0], epsilon, True)

# Vẽ đường viền trên canvas trống
canvas = np.zeros_like(img)
cv2.drawContours(canvas, [approx], -1, (255, 255, 255), thickness=1)

# Lưu hình ảnh đường viền
cv2.imwrite("Mammogram_contour.png", canvas)



# Câu c:
#     Có, một mã chuỗi có thể được sử dụng để đại diện cho đường viền chính trong hình ảnh đường viền. 
# Mã chuỗi là một chuỗi các hướng mô tả đường đi của đường viền. Mỗi hướng được biểu thị bằng một số hoặc chữ cái,
# và chuỗi có thể được sử dụng để tái tạo lại đường viền. Mã chuỗi không bị ảnh hưởng bởi sự dịch chuyển, 
# xoay và tỷ lệ, điều này làm cho nó trở thành một công cụ hữu ích để đại diện cho các đường viền trong các ứng dụng 
# xử lý hình ảnh.

# Trong trường hợp của hình ảnh đường viền xấp xỉ được tạo ra từ hình ảnh nhị phân của “Mammogram.bin”, 
# một mã chuỗi có thể được sử dụng để đại diện cho đường viền chính. Tuy nhiên, vì đường viền không phải 
# là một đường cong đóng, nó sẽ yêu cầu một số xử lý bổ sung để đảm bảo rằng mã chuỗi là hoàn chỉnh và chính xác.