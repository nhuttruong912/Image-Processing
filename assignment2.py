# INT13146 Image Processing (Xu ly Anh)
# Nguyen Phan Nhut Truong N20DCCN082
import cv2

J1 = cv2.imread('lenagray.jpg', cv2.IMREAD_COLOR)

J2 = 255 - J1
cv2.imshow('Photographic Negative', J2)

cv2.imwrite('photographic_negative.jpg', J2)

cv2.waitKey(0)
cv2.destroyAllWindows()
