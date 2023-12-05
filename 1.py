import numpy as np
from scipy.fft import fft2, fftshift

# Define image size
ROWS, COLS = 8, 8

# Create meshgrid for zero-based indexing
COLS, ROWS = np.meshgrid(np.arange(COLS), np.arange(ROWS))

# Given frequencies
u0, v0 = 2, 2

# Define I1
I1 = 0.5 * np.exp(1j * 2 * np.pi * (u0 * COLS + v0 * ROWS) / COLS)

# Display real and imaginary parts of I1 as grayscale images
real_part = np.real(I1)
imag_part = np.imag(I1)

# Print real and imaginary parts as 8x8 ASCII floating point arrays
print("Real part of I1:")
print(np.round(real_part, 4))
print("\nImaginary part of I1:")
print(np.round(imag_part, 4))

# Compute 2D DFT and center it
Itilde1 = fftshift(fft2(I1))

# Print real and imaginary parts of the DFT
print("\nRe[DFT(I1)]:")
print(np.round(np.real(Itilde1), 4))
print("\nIm[DFT(I1)]:")
print(np.round(np.imag(Itilde1), 4))
