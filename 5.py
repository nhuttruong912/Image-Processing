import numpy as np
from scipy.fft import fft2, fftshift
import matplotlib.pyplot as plt

# Define image size
ROWS, COLS = 8, 8

# Create meshgrid for zero-based indexing
COLS, ROWS = np.meshgrid(np.arange(COLS), np.arange(ROWS))

# Given frequencies for I5
u1, v1 = 1.5, 1.5

# Define I5
I5 = np.cos(2 * np.pi * (u1 * COLS + v1 * ROWS) / COLS)

# Display I5 as an 8 bpp grayscale image
plt.imshow(I5, cmap='gray', vmin=-1, vmax=1)
plt.title('I5 as an 8 bpp grayscale image')
plt.colorbar()
plt.show()

# Print real and imaginary parts of I5 as 8x8 ASCII floating point arrays
print("Real part of I5:")
print(np.round(I5, 4))
print("\nImaginary part of I5:")
print(np.zeros_like(I5))  # Since I5 is real, the imaginary part is zero

# Compute 2D DFT of I5 and center it
Itilde5 = fftshift(fft2(I5))

# Print real and imaginary parts of the centered DFT of I5
print("\nRe[DFT(I5)]:")
print(np.round(np.real(Itilde5), 4))
print("\nIm[DFT(I5)]:")
print(np.round(np.imag(Itilde5), 4))
