import cv2
import numpy as np
import sys


# Calculate Sobel Filter for odd dimensions in X and Y edge directions
def calculate_sobel_kernel(target_shape: tuple[int, int]):
    assert target_shape[0] % 2 != 0
    assert target_shape[1] % 2 != 0
    gx = np.zeros(target_shape, dtype=np.float64)
    gy = np.zeros(target_shape, dtype=np.float64)
    indices = np.indices(target_shape, dtype=np.float64)
    cols = indices[0] - target_shape[0] // 2
    rows = indices[1] - target_shape[1] // 2
    squared = cols ** 2 + rows ** 2
    np.divide(cols, squared, out=gy, where=squared != 0)
    np.divide(rows, squared, out=gx, where=squared != 0)
    return gx, gy


# Calculate discrete vals of laplacian of gaussian for odd dimensions
def LoG(sigma, x, y):
    laplace = -1 / (np.pi * sigma ** 4) * (1 - (x ** 2 + y ** 2) / (2 * sigma ** 2)) * np.exp(
        -(x ** 2 + y ** 2) / (2 * sigma ** 2))
    return laplace


def calculate_LoG_kernel(sigma, n):
    # return Laplacian kernel
    if n == 3:
        return np.array([[0, -1, 0],
                         [-1, 4, -1],
                         [0, -1, 0]])
    # return LoG kernel
    else:
        kernel = np.zeros((n, n))
        for i in range(n):
            for j in range(n):
                kernel[i, j] = LoG(sigma, (i - (n - 1) / 2), (j - (n - 1) / 2))
        return kernel


def borders_mirroring(img, filter_size, rows, cols):
    distance = ((filter_size - 1) * 2)
    for i in range(rows):
        for j in range(filter_size - 1):
            img[i][cols - 1 - j] = img[i][cols - distance + j]
    for i in range(cols):
        for j in range(filter_size - 1):
            img[rows - 1 - j][i] = img[rows - distance + j][i]


def smoothing(img, kernel, filter_size):
    print("\nImage Filtering...\n")
    image_rows, image_cols = img.shape[:2]
    kernel_size = kernel.shape[0]

    # Performing the convolution
    convolution = np.zeros((image_rows, image_cols, 3), np.float64)
    for x in range(round((kernel_size - 1) / 2), round(image_rows - ((kernel_size - 1) / 2))):
        for y in range(round((kernel_size - 1) / 2), round(image_cols - ((kernel_size - 1) / 2))):
            sum_1 = 0.0
            sum_2 = 0.0
            sum_3 = 0.0
            for u in range(round(-(kernel_size - 1) / 2), round(((kernel_size - 1) / 2) + 1)):
                for v in range(round(-(kernel_size - 1) / 2), round(((kernel_size - 1) / 2) + 1)):
                    new = kernel[(round(u + ((kernel_size - 1) / 2)), round(v + ((kernel_size - 1) / 2)))]
                    sum_1 = sum_1 + (img[x + u, y + v][0] * new)
                    sum_2 = sum_2 + (img[x + u, y + v][1] * new)
                    sum_3 = sum_3 + (img[x + u, y + v][2] * new)

            i = round(x - ((kernel_size - 1) / 2))
            j = round(y - ((kernel_size - 1) / 2))
            # B Channel
            convolution[i, j][0] = sum_1
            # G Channel
            convolution[i, j][1] = sum_2
            # R Channel
            convolution[i, j][2] = sum_3
    # Performing the pixels mirroring in the borders
    borders_mirroring(convolution, filter_size, image_rows, image_cols)
    return convolution


def edge_detection(img, kernel, filter_size):
    print("\nImage Filtering...\n")
    image_rows, image_cols = img.shape[:2]
    kernel_size = kernel.shape[0]
    #  Performing the convolution
    convolution = np.zeros((image_rows, image_cols), img.dtype)
    for x in range(round((kernel_size - 1) / 2), round(image_rows - ((kernel_size - 1) / 2))):
        for y in range(round((kernel_size - 1) / 2), round(image_cols - ((kernel_size - 1) / 2))):
            sum = 0.0
            for u in range(round(-(kernel_size - 1) / 2), round(((kernel_size - 1) / 2) + 1)):
                for v in range(round(-(kernel_size - 1) / 2), round(((kernel_size - 1) / 2) + 1)):
                    value = kernel[(round(u + ((kernel_size - 1) / 2)), round(v + ((kernel_size - 1) / 2)))]
                    sum = sum + (img[x + u, y + v] * value)

            i = round(x - ((kernel_size - 1) / 2))
            j = round(y - ((kernel_size - 1) / 2))
            convolution[i, j] = sum
    # Performing the pixels mirroring in the borders
    borders_mirroring(convolution, filter_size, image_rows, image_cols)
    return convolution


def smart_threshold(img, i, j, lower, upper):
    if img[i][j] <= lower:
        return 0
    elif img[i][j] >= upper:
        return 255
    else:
        neighbors_dist = int(img.shape[0] // 100)
        if i < img.shape[0] - neighbors_dist and j < img.shape[1] - neighbors_dist:
            # look at the neighbors values
            for x in range(i - neighbors_dist, i + neighbors_dist):
                for y in range(j - neighbors_dist, j + neighbors_dist):
                    if img[x][y] >= upper:
                        return 255
        else:
            return 255
    return 0


def threshold_func(img, lower, upper):
    rows, cols = img.shape[:2]
    threshold_image = np.zeros((rows, cols), img.dtype)
    for i in range(rows):
        for j in range(cols):
            threshold_image[i][j] = smart_threshold(img, i, j, lower, upper)
    return threshold_image


def apply_kernel(image, filter_type, filter_size, kernel):
    filtered = None

    match filter_type:
        case 1:  # Smoothing
            if kernel is None:
                # Custom 2D-Convolution Kernel
                kernel = np.ones((filter_size, filter_size), np.float64) / (filter_size * filter_size)
            # Converting the image to float 64 bits representation
            image = np.float64(image)
            filtered = smoothing(image, kernel, filter_size)
        case 2:  # Edge detection
            if kernel is None:
                edge_det_func = int(input('1 - Gradient Operator (Sobel Filter)\n2 - Laplacian Operator\n----> '))
                if edge_det_func == 1:
                    edge_direction = int(
                        input('1 - Edge enhanced in X-direction\n2 - Edge enhanced in Y-direction\n----> '))
                    Gx, Gy = calculate_sobel_kernel((filter_size, filter_size))
                    if edge_direction == 1:
                        kernel = Gx
                    else:
                        kernel = Gy
                else:
                    sigma = 1.4
                    if filter_size > 3:
                        sigma = float(input('Enter sigma value for the LoG:\n----> '))
                    kernel = calculate_LoG_kernel(sigma, filter_size)

            # Converting the image to Grayscale
            image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            # Converting the image to float 64 bits representation
            image = np.float64(image)
            filtered = edge_detection(image, kernel, filter_size)

            # Threshold Option
            threshold = str(input('Apply threshold (y/n)? \n----> '))
            if threshold == 'y':
                lower = int(input('Enter lower threshold value (0-255, e.g. 40) \n----> '))
                upper = int(input('Enter upper threshold value (0-255, e.g. 120) \n----> '))
                filtered = threshold_func(filtered, lower, upper)
        case _:
            print("Invalid selection")

    return kernel, filtered, image


def display_output(filtered, original_image):
    if filtered is None:
        print('Error. Exiting.')
        sys.exit(0)

    print("\nIn order to proceed,\nPlease close all the images at once (Mouse right click -> close all windows)")

    if len(filtered.shape) > 2:
        filtered = np.uint8(filtered)
    image = np.uint8(original_image)

    diff = cv2.absdiff(image, np.uint8(filtered))
    # Show output images
    cv2.imshow('Original Image = I', image)
    cv2.imshow('Filtered Image = R', filtered)
    cv2.imshow('Difference between I and R', diff)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
    cv2.waitKey(1)


if __name__ == '__main__':
    pathname = sys.argv[1]
    image = cv2.imread(pathname, cv2.IMREAD_COLOR)
    kernel = None

    # Print error message if image is null
    if image is None:
        print('Could not read the image. Exiting.')
        sys.exit(0)

    # Choosing Filter type
    filter_type = int(input('Choose Filter:\n1 - Smoothing\n2 - Edge detection\n----> '))
    while filter_type < 1 or filter_type > 2:
        print("Incorrect number. Try again.")
        filter_type = int(input('Choose Filter:\n1 - Smoothing \n2 - Edge detection\n----> '))

    # Choosing Filter size
    # permitted sizes: 3, 5, 7, 9
    filter_size = int(input('Enter an odd number between 3 to 9 for the size of the kernel:\n----> '))
    while filter_size < 3 or filter_size > 9 or filter_size % 2 == 0:
        print("Incorrect filter size. Try again.")
        filter_size = int(input('Enter an odd number between 3 to 9 for the size of the filter:\n----> '))

    kernel, filtered, float_image = apply_kernel(image, filter_type, filter_size, kernel)
    display_output(filtered, float_image)

    # An option to edit the kernel values
    while True:
        answer = str(
            input('\nDo you want to manually update kernel values and re-display the three images (y/n)?\n----> '))
        if answer == 'y':
            print("Original kernel values for reference:\n", kernel)
            for i in range(filter_size):
                print('Enter the ', i + 1,
                      '-row values for the new kernel and separate them with spaces (e.g. 3 5 22.3):', sep='')
                val = input()
                val = val.split()
                for j in range(filter_size):
                    kernel[i][j] = float(val[j])
            if kernel.sum() != 0:
                # Normalize kernel values to sum of 1
                kernel = kernel / kernel.sum()
            print('\nNew kernel after the normalization:\n ', kernel)
            kernel, filtered, float_image = apply_kernel(image, filter_type, filter_size, kernel)
            display_output(filtered, float_image)
        else:
            break
