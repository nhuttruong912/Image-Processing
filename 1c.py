import numpy as np
import cv2
import matplotlib.pyplot as plt


def contrast_stretch(image):
    stretched_image = (image - np.min(image)) / (np.max(image) - np.min(image))
    return stretched_image


file_path = "salesman.bin"
image_data = np.fromfile(file_path, dtype=np.uint8)


if image_data.size == 0:
    print("Error reading the binary file. Check the file path.")
    exit()


image = image_data.reshape((256, 256))


image = np.pad(image, ((3, 3), (3, 3)), mode='constant', constant_values=0)


image = image.astype(np.float32) / 255.0


filter_window = np.ones((7, 7)) / 49.0
output_image_a = cv2.filter2D(image, -1, filter_window, borderType=cv2.BORDER_CONSTANT)


impulse_response = np.zeros((256, 256))
impulse_response[128-3:128+4, 128-3:128+4] = 1/49.0


zero_padded_impulse_response = np.pad(impulse_response, ((129, 129), (129, 129)), mode='constant')


dft_image = np.fft.fft2(image)
dft_image_shifted = np.fft.fftshift(dft_image)


dft_impulse_response = np.fft.fft2(zero_padded_impulse_response)
dft_impulse_response_shifted = np.fft.fftshift(dft_impulse_response)


dft_image_shifted = dft_image_shifted[:256, :256]
dft_impulse_response_shifted = dft_impulse_response_shifted[:256, :256]


dft_output = dft_image_shifted * dft_impulse_response_shifted

magnitude_spectrum_output = np.log(np.abs(dft_output) + 1)


output_image_c = np.abs(np.fft.ifft2(np.fft.ifftshift(dft_output)))


plt.figure(figsize=(16, 8))

plt.subplot(2, 4, 1)
plt.imshow(contrast_stretch(image), cmap='gray')
plt.title('Original Input Image')

plt.subplot(2, 4, 2)
plt.imshow(contrast_stretch(zero_padded_impulse_response), cmap='gray')
plt.title('Zero-Phase Impulse Response Image')

plt.subplot(2, 4, 3)
plt.imshow(magnitude_spectrum_output, cmap='gray')
plt.title('Magnitude Spectrum of Output Image')

plt.subplot(2, 4, 4)
plt.imshow(contrast_stretch(np.abs(output_image_c)), cmap='gray')
plt.title('Zero Padded Output Image')

plt.show()
