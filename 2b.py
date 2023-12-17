import numpy as np
import matplotlib.pyplot as plt
from scipy.fft import fft2, ifft2, fftshift, ifftshift


def calculate_mse(original, noisy):
    return np.mean((original - noisy) ** 2)


def calculate_isnr(original, noisy, filtered):
    noise = original - noisy
    filtered_noise = original - filtered
    mse_noisy = np.mean(noise ** 2)
    mse_filtered = np.mean(filtered_noise ** 2)
    return 10 * np.log10(mse_noisy / mse_filtered)


def load_binary_image(file_path):
    with open(file_path, 'rb') as f:

        image_data = np.frombuffer(f.read(), dtype=np.uint8)

    return image_data.reshape((256, 256)).astype(float)


file_path_girl2 = 'girl2.bin'
file_path_girl2Noise32Hi = 'girl2Noise32Hi.bin'
file_path_girl2Noise32 = 'girl2Noise32.bin'


girl2 = load_binary_image(file_path_girl2)
girl2Noise32Hi = load_binary_image(file_path_girl2Noise32Hi)
girl2Noise32 = load_binary_image(file_path_girl2Noise32)

plt.figure(figsize=(12, 4))
plt.subplot(1, 3, 1)
plt.imshow(girl2, cmap='gray')
plt.title('Original Image (girl2)')

plt.subplot(1, 3, 2)
plt.imshow(girl2Noise32Hi, cmap='gray')
plt.title('Noisy Image (girl2Noise32Hi)')

plt.subplot(1, 3, 3)
plt.imshow(girl2Noise32, cmap='gray')
plt.title('Noisy Image (girl2Noise32)')

plt.show()


Ucutoff = 64


rows, cols = girl2.shape
u, v = np.meshgrid(np.fft.fftfreq(rows), np.fft.fftfreq(cols))
radius = np.sqrt(u**2 + v**2)
ideal_filter = np.zeros((rows, cols))
ideal_filter[radius <= Ucutoff] = 1


filtered_girl2 = ifft2(fft2(girl2) * ifftshift(ideal_filter))
filtered_girl2Noise32Hi = ifft2(fft2(girl2Noise32Hi) * ifftshift(ideal_filter))
filtered_girl2Noise32 = ifft2(fft2(girl2Noise32) * ifftshift(ideal_filter))


plt.figure(figsize=(12, 4))
plt.subplot(1, 3, 1)
plt.imshow(np.abs(filtered_girl2), cmap='gray')
plt.title('Filtered Image (girl2)')

plt.subplot(1, 3, 2)
plt.imshow(np.abs(filtered_girl2Noise32Hi), cmap='gray')
plt.title('Filtered Image (girl2Noise32Hi)')

plt.subplot(1, 3, 3)
plt.imshow(np.abs(filtered_girl2Noise32), cmap='gray')
plt.title('Filtered Image (girl2Noise32)')

plt.show()


mse_girl2 = calculate_mse(girl2, np.abs(filtered_girl2))
isnr_girl2Noise32Hi = calculate_isnr(girl2, girl2Noise32Hi, np.abs(filtered_girl2Noise32Hi))
isnr_girl2Noise32 = calculate_isnr(girl2, girl2Noise32, np.abs(filtered_girl2Noise32))

print(f"MSE between original girl2 and filtered girl2: {mse_girl2}")
print(f"ISNR between girl2 and girl2Noise32Hi: {isnr_girl2Noise32Hi}")
print(f"ISNR between girl2 and girl2Noise32: {isnr_girl2Noise32}")
