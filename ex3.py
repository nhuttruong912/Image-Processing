# INT13146 Image Processing (Xu ly Anh)
# Nguyen Phan Nhut Truong N20DCCN082

import numpy as np
import matplotlib.pyplot as plt

# Define the image J
COLS, ROWS = np.meshgrid(np.arange(8), np.arange(8))
L = 2
J = np.cos(2 * np.pi * L * (COLS + ROWS) / 8) + 1j * np.sin(2 * np.pi * L * (COLS + ROWS) / 8)
J[0, 0] = 2

# Plot the real and imaginary parts of J as grayscale images
fig, axs = plt.subplots(1, 2)
axs[0].imshow(np.real(J), cmap='gray', vmin=-1, vmax=1)
axs[0].set_title('Real part of J')
axs[1].imshow(np.imag(J), cmap='gray', vmin=-1, vmax=1)
axs[1].set_title('Imaginary part of J')
plt.show()

# Compute the centered DFT of J
Jtilde = np.fft.fftshift(np.fft.fft2(J))

# Print out the real and imaginary parts of Jtilde as an 8 x 8 ascii floating point array
print('Re[DFT(J)]:')
print(np.round(np.real(Jtilde) * 10**4) * 10**(-4))
print('Im[DFT(J)]:')
print(np.round(np.imag(Jtilde) * 10**4) * 10**(-4))
