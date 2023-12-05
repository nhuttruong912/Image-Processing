import cv2
import numpy as np

def approximate_contour(binary_image):
    contours, _ = cv2.findContours(binary_image, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    contour_image = np.zeros_like(binary_image)
    cv2.drawContours(contour_image, contours, -1, 255, thickness=1)
    return contour_image

# Load the binary image obtained by thresholding Mammogram.bin
binary_image_path = "binary_mammogram.bin"
binary_image = np.fromfile(binary_image_path, dtype=np.uint8)
binary_image = binary_image.reshape((256, 256))

# Implement Approximate Contour Image Generation
contour_image = approximate_contour(binary_image)

# Save the contour image (optional)
contour_image_path = "contour_mammogram.bin"
contour_image.tofile(contour_image_path)

# Display the contour image (optional)
cv2.imshow("Contour Image", contour_image)
cv2.waitKey(0)
cv2.destroyAllWindows()
