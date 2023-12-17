import cv2
import numpy as np

def read_binary_image(file_path, width, height):
    with open(file_path, 'rb') as file:
        # Read binary data and convert to NumPy array
        image_data = np.fromfile(file, dtype=np.uint8, count=width * height)
        # Reshape the 1D array to a 2D image array
        image = np.reshape(image_data, (height, width))
        return image

def write_binary_image(file_path, image):
    with open(file_path, 'wb') as file:
        file.write(image.tobytes())

def median_filter(image):
    return cv2.medianBlur(image, 3)

def morphological_opening(image):
    kernel = np.ones((3, 3), np.uint8)
    return cv2.morphologyEx(image, cv2.MORPH_OPEN, kernel)

def morphological_closing(image):
    kernel = np.ones((3, 3), np.uint8)
    return cv2.morphologyEx(image, cv2.MORPH_CLOSE, kernel)

def main():
    # Load binary images
    width, height = 256, 256
    image9 = read_binary_image('camera9.bin', width, height)
    image99 = read_binary_image('camera99.bin', width, height)

    # Apply median filter
    result_median9 = median_filter(image9)
    result_median99 = median_filter(image99)

    # Apply morphological opening
    result_opening9 = morphological_opening(image9)
    result_opening99 = morphological_opening(image99)

    # Apply morphological closing
    result_closing9 = morphological_closing(image9)
    result_closing99 = morphological_closing(image99)

    # Display the images
    cv2.imshow('Original Image 9', image9)
    cv2.imshow('Median Filter 9', result_median9)
    cv2.imshow('Opening 9', result_opening9)
    cv2.imshow('Closing 9', result_closing9)

    cv2.imshow('Original Image 99', image99)
    cv2.imshow('Median Filter 99', result_median99)
    cv2.imshow('Opening 99', result_opening99)
    cv2.imshow('Closing 99', result_closing99)

    cv2.waitKey(0)
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
