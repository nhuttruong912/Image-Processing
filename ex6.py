# INT13146 Image Processing (Xu ly Anh)
# Nguyen Phan Nhut Truong N20DCCN082

import numpy as np
import matplotlib.pyplot as plt

# Load the images
cameraman = plt.imread('cameraman.tif')
salesman = plt.imread('salesman.tif')
bird = plt.imread('bird.gif')
head = plt.imread('head.tif')
eyeblink = plt.imread('eyeblink.gif')

# Define a function to plot an image and its DFT
def plot_dft(image, title):
    # Compute the centered DFT of the image
    Itilde = np.fft.fftshift(np.fft.fft2(image))

    # Plot the original image
    fig, axs = plt.subplots(2, 3)
    axs[0, 0].imshow(image, cmap='gray', vmin=0, vmax=255)
    axs[0, 0].set_title('Original image')

    # Plot the real part of the centered DFT
    axs[0, 1].imshow(np.real(Itilde), cmap='gray', vmin=-10000, vmax=10000)
    axs[0, 1].set_title('Real part of centered DFT')

    # Plot the imaginary part of the centered DFT
    axs[0, 2].imshow(np.imag(Itilde), cmap='gray', vmin=-10000, vmax=10000)
    axs[0, 2].set_title('Imaginary part of centered DFT')

    # Plot the centered DFT log-magnitude spectrum
    axs[1, 1].imshow(np.log10(np.abs(Itilde)), cmap='gray', vmin=0, vmax=5)
    axs[1, 1].set_title('Log-magnitude spectrum of centered DFT')

    # Plot the phase of the centered DFT
    axs[1, 2].imshow(np.angle(Itilde), cmap='gray', vmin=-np.pi, vmax=np.pi)
    axs[1, 2].set_title('Phase of centered DFT')

    # Show the plots
    fig.suptitle(title)
    plt.show()

# Plot each image and its DFT
plot_dft(cameraman, 'Cameraman')
plot_dft(salesman, 'Salesman')
plot_dft(bird, 'Bird')
plot_dft(head, 'Head')
plot_dft(eyeblink, 'Eyeblink')
