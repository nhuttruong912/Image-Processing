# INT13146 Image Processing (Xu ly Anh)
# Nguyen Phan Nhut Truong N20DCCN082

import numpy as np
import matplotlib.pyplot as plt

# Load the image I6
I6 = np.fromfile('camera.bin', dtype=np.uint8).reshape((256, 256))

# Define the images J1 and J2
J1 = np.abs(I6)
J2 = np.angle(I6)

# Plot J2 as an 8 bpp grayscale image with full-scale contrast
plt.imshow(J2, cmap='gray', vmin=-np.pi, vmax=np.pi)
plt.title('J2')
plt.show()

# Plot J1 as an 8 bpp grayscale image with full-scale contrast
JJ1 = np.log(J1)
plt.imshow(JJ1, cmap='gray', vmin=0, vmax=5)
plt.title('JJ1')
plt.show()
