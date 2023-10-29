# INT13146 Image Processing (Xu ly Anh)
# Nguyen Phan Nhut Truong N20DCCN082

import numpy as np
import matplotlib.pyplot as plt

# Define the image I
COLS, ROWS = np.meshgrid(np.arange(8), np.arange(8))
I = 2 * np.pi * np.cos(np.pi * COLS + np.pi * ROWS)

# Plot the real and imaginary parts of I as grayscale images
fig, axs = plt.subplots(1, 2)
axs[0].imshow(I, cmap='gray', vmin=-1, vmax=1)
axs[0].set_title('I')
axs[1].imshow(np.imag(I), cmap='gray', vmin=-1, vmax=1)
axs[1].set_title('Imaginary part of I')
plt.show()

# Compute the centered DFT of I
Itilde = np.fft.fftshift(np.fft.fft2(I))

# Print out the real and imaginary parts of Itilde as an 8 x 8 ascii floating point array
print('Re[DFT(I)]:')
print(np.round(np.real(Itilde) * 10**4) * 10**(-4))
print('Im[DFT(I)]:')
print(np.round(np.imag(Itilde) * 10**4) * 10**(-4))
