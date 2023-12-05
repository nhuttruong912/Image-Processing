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

# Define I3 as the sum of I1 and I2
I3 = I1 + I2

# Display I3 as an 8 bpp grayscale image
plt.imshow(np.real(I3), cmap='gray', vmin=-1, vmax=1)
plt.title('I3 as an 8 bpp grayscale image')
plt.colorbar()
plt.show()

# Print real and imaginary parts of I3 as 8x8 ASCII floating point arrays
print("Real part of I3:")
print(np.round(np.real(I3), 4))
print("\nImaginary part of I3:")
print(np.round(np.imag(I3), 4))

# Compute 2D DFT of I3 and center it
Itilde3 = fftshift(fft2(I3))

# Print real and imaginary parts of the centered DFT of I3
print("\nRe[DFT(I3)]:")
print(np.round(np.real(Itilde3), 4))
print("\nIm[DFT(I3)]:")
print(np.round(np.imag(Itilde3), 4))
