import cv2
import numpy as np
import sys

if __name__ == '__main__':
    image = cv2.imread('test.jpg',cv2.IMREAD_COLOR)

    # Print error message if image is null
    if image is None:
        print('Could not read the image. Exiting.')
        sys.exit(0)
    image_rows, image_cols = image.shape[:2]
    # Converting the image to float 64 bits representation
    image=np.float64(image)

    filter_type = int(input('Choose Filter:\n1 - Smoothing/Blurring \n2 - Edge detection\n----> '))
    while filter_type<1 or filter_type>2:
        print("Incorrect filter type. Try again.")
        filter_type = int(input('Choose Filter:\n1 - Smoothing/Blurring \n2 - Edge detection\n----> '))

    # permitted sizes: 3, 5, 7, 9
    filter_size = int(input('Enter an odd number between 3 to 9 for the size of the filter:\n----> '))


    while filter_size<3 or filter_size>9 or filter_size%2==0:
        print("Incorrect filter size. Try again.")
        filter_size = int(input('Enter an odd number between 3 to 9 for the size of the filter:\n----> '))

    print("Calculation waiting time table:\n\t3x3 Kernel: ~15 Seconds \n\t5x5 Kernel: ~35 Seconds\n\t7x7 Kernel: ~60 Seconds\n\t9x9 Kernel: ~100 Seconds")

    match filter_type:
        case 1: #Smoothing
            kernel = np.ones((filter_size, filter_size), np.float64) / (filter_size * filter_size)
        case 2: #Edge detection
            kernel = np.ones((filter_size, filter_size), np.float64) / (filter_size * filter_size)
        case _:
            print("Invalid selection")


    kernel_rows,kernel_cols=kernel.shape[:2]

    # Adding the countour of nulls around the original image, to avoid border problems during convolution
    img_conv=np.zeros((image_rows + kernel_rows - 1, image_cols + kernel_cols - 1,3), np.float64)
    img_conv_rows, img_conv_cols = img_conv.shape[:2]

    #copying the original image with black borders
    for x in range(image_rows):
        for y in range(image_cols):
            # Channel B
            img_conv[x+1, y+1][0] = image[x, y][0]
            # Channel G
            img_conv[x+1, y+1][1] = image[x, y][1]
            # Channel R
            img_conv[x+1, y+1][2] = image[x, y][2]


    #  Performing the convolution
    my_conv=np.zeros((image_rows, image_cols,3), np.float64)
    for x in range(round((kernel_rows-1) / 2),round(img_conv_rows-((kernel_rows-1) / 2))):
        for y in range(round((kernel_cols - 1) / 2), round(img_conv_cols - ((kernel_cols - 1) / 2))):
            comp_1=0.0
            comp_2=0.0
            comp_3=0.0
            for u in range(round(-(kernel_rows-1) / 2), round(((kernel_rows-1) / 2)+1)):
                for v in range(round(-(kernel_cols - 1) / 2), round(((kernel_cols - 1) / 2) + 1)):
                    new=kernel[(round(u + ((kernel_rows-1) / 2)), round(v + ((kernel_cols-1) / 2)))]
                    comp_1 = comp_1 + (img_conv[x+u, y+v][0] * new)
                    comp_2 = comp_2 + (img_conv[x+u, y+v][1] * new)
                    comp_3 = comp_3 + (img_conv[x+u, y+v][2] * new)

            i=round(x-((kernel_rows-1) / 2))
            j=round(y-((kernel_cols-1) / 2))
            my_conv[i,j][0] = comp_1
            my_conv[i,j][1] = comp_2
            my_conv[i,j][2] = comp_3
    print("finished")
    # Converting the output image back to uint 8 bits representation
    my_conv=np.uint8(my_conv)
    # We should get the same image
    cv2.imshow('Original', my_conv)

    cv2.waitKey(0)
    cv2.destroyAllWindows()


