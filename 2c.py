import numpy as np
import matplotlib.pyplot as plt
from scipy.fft import fft2, ifft2, fftshift, ifftshift, fftn


def calculate_mse(original, noisy):
    return np.mean((original - noisy) ** 2)


def calculate_isnr(original, noisy, filtered):
    noise = original - noisy
    filtered_noise = original - filtered
    mse_noisy = np.mean(noise ** 2)
    mse_filtered = np.mean(filtered_noise ** 2)
    return 10 * np.log10(mse_noisy / mse_filtered)


def gaussian_low_pass_filter(N, Ucutoff):
    sigma = 0.19 * N / Ucutoff
    [U, V] = np.meshgrid(np.fft.fftfreq(N), np.fft.fftfreq(N))
    HtildeCenter = np.exp((-2 * np.pi**2 * sigma**2) / (N**2) * (U**2 + V**2))
    Htilde = fftshift(HtildeCenter)
    H = ifft2(ifftshift(Htilde))
    H_shifted = fftshift(H)
    return H_shifted


def load_binary_image(file_path, image_size=(256, 256)):
    with open(file_path, 'rb') as f:

        image_data = np.frombuffer(f.read(), dtype=np.uint8)

    return image_data.reshape(image_size).astype(float)


file_path_girl2 = 'girl2.bin'
file_path_girl2Noise32Hi = 'girl2Noise32Hi.bin'
file_path_girl2Noise32 = 'girl2Noise32.bin'


girl2 = load_binary_image(file_path_girl2)
girl2Noise32Hi = load_binary_image(file_path_girl2Noise32Hi)
girl2Noise32 = load_binary_image(file_path_girl2Noise32)


Ucutoff = 64


H_gaussian = gaussian_low_pass_filter(N=256, Ucutoff=Ucutoff)

filtered_girl2 = ifft2(fft2(girl2) * fftn(ifftshift(H_gaussian)))
filtered_girl2Noise32Hi = ifft2(fft2(girl2Noise32Hi) * fftn(ifftshift(H_gaussian)))
filtered_girl2Noise32 = ifft2(fft2(girl2Noise32) * fftn(ifftshift(H_gaussian)))


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
