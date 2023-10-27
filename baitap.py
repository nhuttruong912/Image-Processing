import cv2
import numpy as np
import matplotlib.pyplot as plt

# Load the image
img = cv2.imread('parrot.jpg')

# Convert the image to HSV color space
hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)

# Define the range of colors to segment (e.g., green objects)
lower_color = np.array([40, 50, 50])
upper_color = np.array([90, 255, 255])

# Create a binary mask where pixels within the color range are white and others are black
mask = cv2.inRange(hsv, lower_color, upper_color)

# Apply morphological operations to clean up the mask
kernel = np.ones((5,5),np.uint8)
mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)

# Find contours in the mask
contours, hierarchy = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

# Draw bounding boxes around each object
for cnt in contours:
    x,y,w,h = cv2.boundingRect(cnt)
    cv2.rectangle(img,(x,y),(x+w,y+h),(0,255,0),3)

# Display the segmented image with bounding boxes
plt.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
plt.show()
