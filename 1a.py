import numpy as np
import cv2
import matplotlib.pyplot as plt


file_path = "salesman.bin"
image_data = np.fromfile(file_path, dtype=np.uint8)


if image_data.size == 0:
    print("Error reading the binary file. Check the file path.")
    exit()


image = image_data.reshape((256, 256))


image = np.pad(image, ((3, 3), (3, 3)), mode='constant', constant_values=0)


image = image.astype(np.float32) / 255.0


output_image = np.zeros_like(image)


filter_window = np.ones((7, 7)) / 49.0


for i in range(3, image.shape[0] - 3):
    for j in range(3, image.shape[1] - 3):
        window = image[i-3:i+4, j-3:j+4]
        output_image[i, j] = np.sum(window * filter_window)


plt.figure(figsize=(10, 5))

plt.subplot(1, 2, 1)
plt.imshow(image[3:-3, 3:-3], cmap='gray')
plt.title('Input Image')

plt.subplot(1, 2, 2)
plt.imshow(output_image[3:-3, 3:-3], cmap='gray')
plt.title('Output Image')

plt.show()
