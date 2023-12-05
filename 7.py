import numpy as np
from scipy.fft import fft2, ifft2, fftshift
import matplotlib.pyplot as plt

# Assuming you have a function to load the image
def load_image(file_path):
    with open(file_path, 'rb') as file:
        image = np.fromfile(file, dtype=np.uint8, count=256*256)
    return image.reshape((256, 256))

# Load the Cameraman image from camera.bin
I6 = load_image("camera.bin")

# Compute the DFT of I6
I6_dft = fftshift(fft2(I6))

# Create J1 and J2
J1 = np.abs(I6)
J2 = np.exp(1j * np.angle(I6))

# Display J2 as an 8 bpp grayscale image with full-scale contrast
plt.imshow(np.real(J2), cmap='gray', vmin=-1, vmax=1)
plt.title('J2 as an 8 bpp grayscale image')
plt.colorbar()
plt.show()

# Create JJ1 by taking the log of J1
JJ1 = np.log(J1)

# Display JJ1 as an 8 bpp grayscale image with full-scale contrast
plt.imshow(JJ1, cmap='gray')
plt.title('JJ1 as an 8 bpp grayscale image')
plt.colorbar()
plt.show()
