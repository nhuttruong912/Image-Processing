import numpy as np
from scipy.fft import fft2, fftshift
import matplotlib.pyplot as plt

# Define image size
ROWS, COLS = 8, 8

# Create meshgrid for zero-based indexing
COLS, ROWS = np.meshgrid(np.arange(COLS), np.arange(ROWS))

# Given frequencies
u0, v0 = 2, 2

# Define I2
I2 = 1 * np.exp(-1j * 2 * np.pi * (u0 * COLS + v0 * ROWS) / COLS)

# Display real and imaginary parts of I2 as grayscale images
real_part = np.real(I2)
imag_part = np.imag(I2)

# Plot real and imaginary parts
plt.figure(figsize=(8, 4))

plt.subplot(1, 2, 1)
plt.imshow(real_part, cmap='gray', vmin=-1, vmax=1)
plt.title('Real part of I2')
plt.colorbar()

plt.subplot(1, 2, 2)
plt.imshow(imag_part, cmap='gray', vmin=-1, vmax=1)
plt.title('Imaginary part of I2')
plt.colorbar()

plt.show()

# Print real and imaginary parts as 8x8 ASCII floating point arrays
print("Real part of I2:")
print(np.round(real_part, 4))
print("\nImaginary part of I2:")
print(np.round(imag_part, 4))

# Compute 2D DFT and center it
Itilde2 = fftshift(fft2(I2))

# Print real and imaginary parts of the DFT
print("\nRe[DFT(I2)]:")
print(np.round(np.real(Itilde2), 4))
print("\nIm[DFT(I2)]:")
print(np.round(np.imag(Itilde2), 4))
