import numpy as np
from scipy.fft import fft2, fftshift
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm

# Function to load image from binary file
def load_image(file_path):
    with open(file_path, 'rb') as file:
        image = np.fromfile(file, dtype=np.uint8, count=256*256)
    return image.reshape((256, 256))

# Function to display an image
def display_image(image, title):
    plt.imshow(image, cmap='gray', vmin=0, vmax=255)
    plt.title(title)
    plt.colorbar()
    plt.show()

# Function to display DFT components
def display_dft(image, title):
    dft = fftshift(fft2(image))

    # Display real part
    display_image(np.real(dft), f"Real part of {title} DFT")

    # Display imaginary part
    display_image(np.imag(dft), f"Imaginary part of {title} DFT")

    # Display log-magnitude spectrum
    magnitude_spectrum = np.log(np.abs(dft) + 1)  # Add 1 to avoid log(0)
    display_image(magnitude_spectrum, f"Log-Magnitude Spectrum of {title} DFT")

    # Display phase
    phase = np.angle(dft)
    display_image(phase, f"Phase of {title} DFT")

# Load images
image_files = ["camera.bin", "salesman.bin", "head.bin", "eyeR.bin"]

for image_file in image_files:
    # Load the image using the defined function
    image = load_image(image_file)

    # Display original image
    display_image(image, f"Original {image_file}")

    # Display DFT
    display_dft(image, image_file)
