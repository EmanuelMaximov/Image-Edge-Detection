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
    laplace = -1/(np.pi*sigma**4)*(1-(x**2+y**2)/(2*sigma**2))*np.exp(-(x**2+y**2)/(2*sigma**2))
    return laplace

def calculate_LoG_kernel(sigma, n):
    l = np.zeros((n,n))
    for i in range(n):
        for j in range(n):
            l[i,j] = LoG(sigma, (i-(n-1)/2),(j-(n-1)/2))
    return l


def smoothing(img, kernel):
    print("Applying kernel on the image...\n")
    image_rows, image_cols = img.shape[:2]
    kernel_rows, kernel_cols = kernel.shape[:2]

    # Adding the contour of nulls around the original image, to avoid border problems during convolution
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


    # Performing the convolution
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

def threshold_func(val,th):
    if val<th:
        return 0
    else:
        return 255

def edge_detection(img, kernel,threshold):
    print("Applying kernel on the image...\n")
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
            if threshold:
                comp_1=threshold_func(comp_1,100)
            my_conv[i, j] = comp_1
    return my_conv

def apply_kernel(image,filter_type,filter_size,kernel):
    filtered=None

    match filter_type:
        case 1:  # Smoothing
            if kernel is None:
                kernel = np.ones((filter_size, filter_size), np.float64) / (filter_size * filter_size)
            # Converting the image to float 64 bits representation
            image = np.float64(image)
            filtered = smoothing(image, kernel)
        case 2:  # Edge detection
            if kernel is None:
                edge_det_func = int(input('1 - Gradient Operator (Sobel Filter)\n2 - Laplacian Operator\n----> '))
                if edge_det_func==1:
                    edge_direction = int(input('1 - Edge enhanced in X-direction\n2 - Edge enhanced in Y-direction\n----> '))
                    if edge_direction==1:
                        kernel = calculate_sobel_kernel((filter_size, filter_size))[0]
                    else:
                        kernel = calculate_sobel_kernel((filter_size, filter_size))[1]
                else:
                    kernel=calculate_LoG_kernel(1.4,filter_size)
            threshold = str(input('Apply threshold (y/n)? \n----> '))
            if threshold == 'y':
                threshold = True
            else:
                threshold = False
            image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            # Converting the image to float 64 bits representation
            image = np.float64(image)
            filtered = edge_detection(image, kernel,threshold)
        case _:
            print("Invalid selection")

    if filtered is None:
        print('Error. Exiting.')
        sys.exit(0)

    print("Please close all windows at once")

    if len(filtered.shape) > 2:
        filtered = np.uint8(filtered)
    image = np.uint8(image)
    diff = cv2.absdiff(image, np.uint8(filtered))

    # Show output images
    cv2.imshow('Original', image)
    cv2.imshow('Filtered Image', filtered)
    cv2.imshow('Diff', diff)
    cv2.waitKeyEx(0)
    # # save the output in the PC
    cv2.imwrite("Filtered.jpg", filtered)
    cv2.destroyAllWindows()
    cv2.waitKey(1)
    return kernel


if __name__ == '__main__':
    image = cv2.imread('test4.jpg', cv2.IMREAD_COLOR)
    kernel = None
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

    # print(
    #     "Calculation time table:\n\t3x3 Kernel: ~15 Seconds \n\t5x5 Kernel: ~35 Seconds\n\t7x7 Kernel: ~60 Seconds\n\t9x9 Kernel: ~100 Seconds")
    kernel=apply_kernel(image, filter_type, filter_size,kernel)

    while True:
        answer = str(input('Do you want to manually change the kernel (y/n)?\n----> '))
        if answer == 'y':
            print("Original kernel values:\n",kernel)
            for i in range(filter_size):
                for j in range(filter_size):
                    print('Enter the', [i,j], 'value for the new kernel:')
                    kernel[i][j] = float((input()))
            if kernel.sum()!=0:
                # Normalize kernel values to sum of 1
                kernel=kernel/kernel.sum()
            print('New kernel after the normalization:\n ',kernel)
            apply_kernel(image, filter_type, filter_size, kernel)
        else:
            break


