# INT13146 Image Processing (Xu ly Anh)
# Nguyen Phan Nhut Truong N20DCCN082
import cv2

J1 = cv2.imread("lena512color.jpg")

cv2.imshow("Original Image (J1)", J1)
cv2.waitKey(0)

J2 = J1.copy()
J2[:, :, 0] = J1[:, :, 2]
J2[:, :, 1] = J1[:, :, 0]
J2[:, :, 2] = J1[:, :, 1]

cv2.imshow("Modified Image (J2)", J2)
cv2.waitKey(0)

cv2.imwrite("modified_lena.jpg", J2)

cv2.destroyAllWindows()

cv2.imshow("Red Band of J1", J1[:, :, 2])
cv2.imshow("Green Band of J1", J1[:, :, 1])
cv2.imshow("Blue Band of J1", J1[:, :, 0])

cv2.imshow("Red Band of J2", J2[:, :, 0])
cv2.imshow("Green Band of J2", J2[:, :, 1])
cv2.imshow("Blue Band of J2", J2[:, :, 2])

cv2.imwrite("red_band_J2.jpg", J2[:, :, 0])
cv2.imwrite("green_band_J2.jpg", J2[:, :, 1])
cv2.imwrite("blue_band_J2.jpg", J2[:, :, 2])

cv2.waitKey(0)
cv2.destroyAllWindows()
