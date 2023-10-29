# INT13146 Image Processing (Xu ly Anh)
# Nguyen Phan Nhut Truong N20DCCN082

import numpy as np
import matplotlib.pyplot as plt

# Define the image I
COLS, ROWS = np.meshgrid(np.arange(8), np.arange(8))
w0 = 2 * np.pi / 8
I = 1 - 2 ** (-COLS) - 2 ** (-ROWS)
I *= np.exp(1j * w0 * (COLS + ROWS))

# Plot the real and imaginary parts of I as grayscale images
fig, axs = plt.subplots(1, 2)
axs[0].imshow(np.real(I), cmap='gray', vmin=-1, vmax=1)
axs[0].set_title('Real part of I')
axs[1].imshow(np.imag(I), cmap='gray', vmin=-1, vmax=1)
axs[1].set_title('Imaginary part of I')
plt.show()

# Compute the centered DFT of I
Itilde = np.fft.fftshift(np.fft.fft2(I))

# Print out the real and imaginary parts of Itilde as 8 × 8 ascii floating point arrays
print('Re[DFT(I)]:')
print(np.round(np.real(Itilde) * 10**4) * 10**(-4))
print('Im[DFT(I)]:')
print(np.round(np.imag(Itilde) * 10**4) * 10**(-4))
