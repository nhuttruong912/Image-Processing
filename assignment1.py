# INT13146 Image Processing (Xu ly Anh)
# Nguyen Phan Nhut Truong N20DCCN082

import numpy as np
import matplotlib.pyplot as plt

lena = np.fromfile("lena.bin", dtype=np.uint8).reshape((256, 256))
peppers = np.fromfile("peppers.bin", dtype=np.uint8).reshape((256, 256))

plt.figure(figsize=(10, 5))
plt.subplot(121)
plt.title("Lena")
plt.imshow(lena, cmap='gray')

plt.subplot(122)
plt.title("Peppers")
plt.imshow(peppers, cmap='gray')
plt.show()

J = np.zeros((256, 256), dtype=np.uint8)
J[:, :128] = lena[:, :128]
J[:, 128:] = peppers[:, 128:]

plt.figure()
plt.title("Image J")
plt.imshow(J, cmap='gray')
plt.show()

K = np.copy(J)
K[:, :128] = J[:, 128:]
K[:, 128:] = J[:, :128]

plt.figure()
plt.title("Image K")
plt.imshow(K, cmap='gray')
plt.show()
