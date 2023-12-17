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


zero_padded_image = np.pad(image, ((128, 128), (128, 128)), mode='constant')


impulse_response = np.zeros((128, 128))
impulse_response[64-3:64+4, 64-3:64+4] = 1/49.0


zero_padded_impulse_response = np.pad(impulse_response, ((128, 128), (128, 128)), mode='constant')


dft_image = np.fft.fft2(zero_padded_image)
dft_image_shifted = np.fft.fftshift(dft_image)
magnitude_spectrum_image = np.log(np.abs(dft_image_shifted) + 1)


dft_impulse_response = np.fft.fft2(zero_padded_impulse_response)
dft_impulse_response_shifted = np.fft.fftshift(dft_impulse_response)
magnitude_spectrum_impulse_response = np.log(np.abs(dft_impulse_response_shifted) + 1)


rows, cols = zero_padded_image.shape
dft_impulse_response_shifted = np.pad(dft_impulse_response_shifted, ((rows//2, rows//2), (cols//2, cols//2)), mode='constant')


dft_impulse_response_shifted = dft_impulse_response_shifted[:rows, :cols]


dft_output = dft_image * dft_impulse_response_shifted


magnitude_spectrum_output = np.log(np.abs(np.fft.fftshift(dft_output)) + 1)


output_image_b = np.abs(np.fft.ifft2(np.fft.ifftshift(dft_output)))


plt.figure(figsize=(16, 12))

plt.subplot(3, 4, 1)
plt.imshow(contrast_stretch(image), cmap='gray')
plt.title('Original Input Image')

plt.subplot(3, 4, 2)
plt.imshow(contrast_stretch(zero_padded_image), cmap='gray')
plt.title('Zero Padded Original Image')

plt.subplot(3, 4, 3)
plt.imshow(contrast_stretch(zero_padded_impulse_response), cmap='gray')
plt.title('Zero Padded Impulse Response Image')

plt.subplot(3, 4, 4)
plt.imshow(magnitude_spectrum_image, cmap='gray')
plt.title('Magnitude Spectrum of Input Image')

plt.subplot(3, 4, 5)
plt.imshow(magnitude_spectrum_impulse_response, cmap='gray')
plt.title('Magnitude Spectrum of Impulse Response')

plt.subplot(3, 4, 6)
plt.imshow(magnitude_spectrum_output, cmap='gray')
plt.title('Magnitude Spectrum of Output Image')

plt.subplot(3, 4, 7)
plt.imshow(contrast_stretch(np.abs(output_image_b)), cmap='gray')
plt.title('Zero Padded Output Image')


final_output_image = np.abs(output_image_b[128:384, 128:384])
plt.subplot(3, 4, 8)
plt.imshow(contrast_stretch(final_output_image), cmap='gray')
plt.title('Final 256x256 Output Image')

plt.show()
