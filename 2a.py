import numpy as np
import matplotlib.pyplot as plt


def calculate_mse(original, noisy):
    return np.mean((original - noisy) ** 2)


file_path_girl2 = "girl2.bin"
file_path_girl2Noise32Hi = "girl2Noise32Hi.bin"
file_path_girl2Noise32 = "girl2Noise32.bin"


with open(file_path_girl2, 'rb') as f:
    girl2_data = np.frombuffer(f.read(), dtype=np.uint8)

with open(file_path_girl2Noise32Hi, 'rb') as f:
    girl2Noise32Hi_data = np.frombuffer(f.read(), dtype=np.uint8)

with open(file_path_girl2Noise32, 'rb') as f:
    girl2Noise32_data = np.frombuffer(f.read(), dtype=np.uint8)

if girl2_data.size == 0 or girl2Noise32Hi_data.size == 0 or girl2Noise32_data.size == 0:
    print("Error reading the binary files. Check the file paths.")
    exit()


girl2 = girl2_data.reshape((256, 256))
girl2Noise32Hi = girl2Noise32Hi_data.reshape((256, 256))
girl2Noise32 = girl2Noise32_data.reshape((256, 256))


plt.figure(figsize=(12, 4))

plt.subplot(1, 3, 1)
plt.imshow(girl2, cmap='gray')
plt.title('Original Image (girl2)')

plt.subplot(1, 3, 2)
plt.imshow(girl2Noise32Hi, cmap='gray')
plt.title('Noisy Image (girl2Noise32Hi)')

plt.subplot(1, 3, 3)
plt.imshow(girl2Noise32, cmap='gray')
plt.title('Noisy Image (girl2Noise32)')

plt.show()


mse_girl2Noise32Hi = calculate_mse(girl2, girl2Noise32Hi)
mse_girl2Noise32 = calculate_mse(girl2, girl2Noise32)

print(f"MSE between girl2 and girl2Noise32Hi: {mse_girl2Noise32Hi}")
print(f"MSE between girl2 and girl2Noise32: {mse_girl2Noise32}")
