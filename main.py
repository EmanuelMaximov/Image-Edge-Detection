import cv2
import numpy as np
import sys


def calc_sobel_kernel(target_shape: tuple[int, int]):
    assert target_shape[0] % 2 != 0
    assert target_shape[1] % 2 != 0
    gx = np.zeros(target_shape, dtype=np.float32)
    gy = np.zeros(target_shape, dtype=np.float32)
    indices = np.indices(target_shape, dtype=np.float32)
    cols = indices[0] - target_shape[0] // 2
    rows = indices[1] - target_shape[1] // 2
    squared = cols ** 2 + rows ** 2
    np.divide(cols, squared, out=gy, where=squared != 0)
    np.divide(rows, squared, out=gx, where=squared != 0)
    return gx, gy


def smoothing(img, kernel):
    image_rows, image_cols = img.shape[:2]
    kernel_rows, kernel_cols = kernel.shape[:2]

    # Adding the countour of nulls around the original image, to avoid border problems during convolution
    img_conv = np.zeros((image_rows + kernel_rows - 1, image_cols + kernel_cols - 1, 3), np.float64)
    img_conv_rows, img_conv_cols = img_conv.shape[:2]

    # copying the original image with black borders
    for x in range(image_rows):
        for y in range(image_cols):
            # Channel B
            img_conv[x + 1, y + 1][0] = img[x, y][0]
            # Channel G
            img_conv[x + 1, y + 1][1] = img[x, y][1]
            # Channel R
            img_conv[x + 1, y + 1][2] = img[x, y][2]
    # copying the original image with black borders

    #  Performing the convolution
    my_conv = np.zeros((image_rows, image_cols, 3), np.float64)
    for x in range(round((kernel_rows - 1) / 2), round(img_conv_rows - ((kernel_rows - 1) / 2))):
        for y in range(round((kernel_cols - 1) / 2), round(img_conv_cols - ((kernel_cols - 1) / 2))):
            comp_1 = 0.0
            comp_2 = 0.0
            comp_3 = 0.0
            for u in range(round(-(kernel_rows - 1) / 2), round(((kernel_rows - 1) / 2) + 1)):
                for v in range(round(-(kernel_cols - 1) / 2), round(((kernel_cols - 1) / 2) + 1)):
                    new = kernel[(round(u + ((kernel_rows - 1) / 2)), round(v + ((kernel_cols - 1) / 2)))]
                    comp_1 = comp_1 + (img_conv[x + u, y + v][0] * new)
                    comp_2 = comp_2 + (img_conv[x + u, y + v][1] * new)
                    comp_3 = comp_3 + (img_conv[x + u, y + v][2] * new)

            i = round(x - ((kernel_rows - 1) / 2))
            j = round(y - ((kernel_cols - 1) / 2))
            my_conv[i, j][0] = comp_1
            my_conv[i, j][1] = comp_2
            my_conv[i, j][2] = comp_3
    return my_conv


def edge_detection(img, kernel):
    image_rows, image_cols = img.shape[:2]
    kernel_rows, kernel_cols = kernel.shape[:2]
    img_conv = np.zeros((image_rows + kernel_rows - 1, image_cols + kernel_cols - 1), img.dtype)
    img_conv_rows, img_conv_cols = img_conv.shape[:2]

    for x in range(image_rows):
        for y in range(image_cols):
            img_conv[x + 1, y + 1] = img[x, y]

    #  Performing the convolution
    my_conv = np.zeros((image_rows, image_cols), img.dtype)
    for x in range(round((kernel_rows - 1) / 2), round(img_conv_rows - ((kernel_rows - 1) / 2))):
        for y in range(round((kernel_cols - 1) / 2), round(img_conv_cols - ((kernel_cols - 1) / 2))):
            comp_1 = 0.0
            for u in range(round(-(kernel_rows - 1) / 2), round(((kernel_rows - 1) / 2) + 1)):
                for v in range(round(-(kernel_cols - 1) / 2), round(((kernel_cols - 1) / 2) + 1)):
                    new = kernel[(round(u + ((kernel_rows - 1) / 2)), round(v + ((kernel_cols - 1) / 2)))]
                    comp_1 = comp_1 + (img_conv[x + u, y + v] * new)

            i = round(x - ((kernel_rows - 1) / 2))
            j = round(y - ((kernel_cols - 1) / 2))
            my_conv[i, j] = comp_1
    return my_conv


if __name__ == '__main__':
    image = cv2.imread('test.png', cv2.IMREAD_COLOR)
    filtered=None

    # Print error message if image is null
    if image is None:
        print('Could not read the image. Exiting.')
        sys.exit(0)

    filter_type = int(input('Choose Filter:\n1 - Smoothing/Blurring \n2 - Edge detection\n----> '))
    while filter_type < 1 or filter_type > 2:
        print("Incorrect filter type. Try again.")
        filter_type = int(input('Choose Filter:\n1 - Smoothing/Blurring \n2 - Edge detection\n----> '))

    # permitted sizes: 3, 5, 7, 9
    filter_size = int(input('Enter an odd number between 3 to 9 for the size of the filter:\n----> '))

    while filter_size < 3 or filter_size > 9 or filter_size % 2 == 0:
        print("Incorrect filter size. Try again.")
        filter_size = int(input('Enter an odd number between 3 to 9 for the size of the filter:\n----> '))

    print(
        "Calculation waiting time table:\n\t3x3 Kernel: ~15 Seconds \n\t5x5 Kernel: ~35 Seconds\n\t7x7 Kernel: ~60 Seconds\n\t9x9 Kernel: ~100 Seconds")

    match filter_type:
        case 1:  # Smoothing
            kernel = np.ones((filter_size, filter_size), np.float64) / (filter_size * filter_size)
            # Converting the image to float 64 bits representation
            image = np.float64(image)
            filtered=smoothing(image,kernel)
        case 2:  # Edge detection
            kernel = calc_sobel_kernel((filter_size, filter_size))[0]
            image= cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            # Converting the image to float 64 bits representation
            image = np.float64(image)
            filtered=edge_detection(image, kernel)
        case _:
            print("Invalid selection")

    if filtered is None:
        print('Error. Exiting.')
        sys.exit(0)
    print("Done")

    if  len(filtered.shape)>2:
        filtered=np.uint8(filtered)
    image = np.uint8(image)
    diff = cv2.absdiff(image, np.uint8(filtered))
    cv2.imshow('Original', image)
    cv2.imshow('Filtered Image', filtered)
    cv2.imshow('Diff', diff)
    cv2.imwrite("new.jpg", filtered)
    cv2.waitKey(0)
    cv2.destroyAllWindows()



