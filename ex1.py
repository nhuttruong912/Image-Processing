# INT13146 Image Processing (Xu ly Anh)
# Nguyen Phan Nhut Truong N20DCCN082

import numpy as np
import matplotlib.pyplot as plt

# Define the image I1
COLS, ROWS = np.meshgrid(np.arange(8), np.arange(8))
u0, v0 = 2, 2
I1 = np.exp(1j * 2 * np.pi * (u0 * COLS + v0 * ROWS) / 8)

# Plot the real and imaginary parts of I1 as grayscale images
fig, axs = plt.subplots(1, 2)
axs[0].imshow(np.real(I1), cmap='gray', vmin=-1, vmax=1)
axs[0].set_title('Real part of I1')
axs[1].imshow(np.imag(I1), cmap='gray', vmin=-1, vmax=1)
axs[1].set_title('Imaginary part of I1')
plt.show()

# Compute the DFT of I1 and center it
Itilde1 = np.fft.fftshift(np.fft.fft2(I1))

# Print out the real and imaginary parts of Itilde1 as 8 × 8 ascii floating point arrays
print('Re[DFT(I1)]:')
print(np.round(np.real(Itilde1) * 10**4) * 10**(-4))
print('Im[DFT(I1)]:')
print(np.round(np.imag(Itilde1) * 10**4) * 10**(-4))
