import numpy as np
from scipy.fft import fft2, fftshift
import matplotlib.pyplot as plt

# Define image size
ROWS, COLS = 8, 8

# Create meshgrid for zero-based indexing
COLS, ROWS = np.meshgrid(np.arange(COLS), np.arange(ROWS))

# Given frequencies
u0, v0 = 2, 2

# Define I1
I1 = 0.5 * np.exp(1j * 2 * np.pi * (u0 * COLS + v0 * ROWS) / COLS)

# Define I2
I2 = 1 * np.exp(-1j * 2 * np.pi * (u0 * COLS + v0 * ROWS) / COLS)

# Define I4 as -j(I1 - I2)
I4 = -1j * (I1 - I2)

# Display I4 as an 8 bpp grayscale image
plt.imshow(np.imag(I4), cmap='gray', vmin=-1, vmax=1)
plt.title('I4 as an 8 bpp grayscale image')
plt.colorbar()
plt.show()

# Print real and imaginary parts of I4 as 8x8 ASCII floating point arrays
print("Real part of I4:")
print(np.round(np.real(I4), 4))
print("\nImaginary part of I4:")
print(np.round(np.imag(I4), 4))

# Compute 2D DFT of I4 and center it
Itilde4 = fftshift(fft2(I4))

# Print real and imaginary parts of the centered DFT of I4
print("\nRe[DFT(I4)]:")
print(np.round(np.real(Itilde4), 4))
print("\nIm[DFT(I4)]:")
print(np.round(np.imag(Itilde4), 4))
