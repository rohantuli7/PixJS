import random
import numpy as np
import matplotlib.pyplot as plt

def correlation_analysis(orig, cipher, img_name):
    max_x_axis = orig.shape[0] - 2
    max_y_axis = orig.shape[1] - 2
    x_axis = [random.randint(0, max_x_axis) for i in range(10000)]
    y_axis = [random.randint(0, max_y_axis) for i in range(10000)]
    path_horizontal = '/Users/rt/Desktop/College/projects/PixJS/readings/correlation/horizontal'
    path_vertical = '/Users/rt/Desktop/College/projects/PixJS/readings/correlation/vertical'
    path_diagonal = '/Users/rt/Desktop/College/projects/PixJS/readings/correlation/diagonal'

    plain_image_values = [orig[i][j] for i, j in zip(x_axis, y_axis)]
    cipher_image_values = [cipher[i][j] for i, j in zip(x_axis, y_axis)]

    #horizontal correlation calculation
    y_axis_horizontal = [i + 1 for i in y_axis]

    plain_image_horizontal = [orig[i][j] for i, j in zip(x_axis, y_axis_horizontal)]
    plt.scatter(plain_image_values, plain_image_horizontal, s = 0.3, color = 'skyblue')
    plt.title(f'{img_name} horizontal direction of plain image')
    plt.xlabel('Pixel gray value on location (x, y)')
    plt.ylabel('Pixel gray value on location (x+1, y)')
    plt.savefig(f'{path_horizontal}/{img_name}_plain_image.png')
    plt.show()

    cipher_image_horizontal = [cipher[i][j] for i, j in zip(x_axis, y_axis_horizontal)]
    plt.scatter(cipher_image_values, cipher_image_horizontal, s = 0.3, color = 'darkblue')
    plt.title(f'{img_name} horizontal direction of cipher image')
    plt.xlabel('Pixel gray value on location (x, y)')
    plt.ylabel('Pixel gray value on location (x+1, y)')
    plt.savefig(f'{path_horizontal}/{img_name}_cipher_image.png')
    plt.show()

    #vertical correlation calculation
    x_axis_vertical = [i + 1 for i in x_axis]

    plain_image_vertical = [orig[i][j] for i, j in zip(x_axis_vertical, y_axis)]
    plt.scatter(plain_image_values, plain_image_vertical, s = 0.3, color = 'skyblue')
    plt.title(f'{img_name} vertical direction of plain image')
    plt.xlabel('Pixel gray value on location (x, y)')
    plt.ylabel('Pixel gray value on location (x, y+1)')
    plt.savefig(f'{path_vertical}/{img_name}_plain_image.png')
    plt.show()

    cipher_image_vertical = [cipher[i][j] for i, j in zip(x_axis_vertical, y_axis)]
    plt.scatter(cipher_image_values, cipher_image_vertical, s = 0.3, color = 'darkblue')
    plt.title(f'{img_name} vertical direction of cipher image')
    plt.xlabel('Pixel gray value on location (x, y)')
    plt.ylabel('Pixel gray value on location (x, y+1)')
    plt.savefig(f'{path_vertical}/{img_name}_cipher_image.png')
    plt.show()


    #diagonal correlation calculation
    plain_image_diagonal = [orig[i][j] for i, j in zip(x_axis_vertical, y_axis_horizontal)]
    plt.scatter(plain_image_diagonal, plain_image_values, s = 0.3, color = 'skyblue')
    plt.title(f'{img_name} diagonal direction of the plain image')
    plt.xlabel('Pixel gray value on location (x, y)')
    plt.ylabel('Pixel gray value on location (x+1, y+1)')
    plt.savefig(f'{path_diagonal}/{img_name}_plain_image.png')
    plt.show()

    cipher_image_diagonal = [cipher[i][j] for i, j in zip(x_axis_vertical, y_axis_horizontal)]
    plt.scatter(cipher_image_values, cipher_image_diagonal, s = 0.3, color = 'darkblue')
    plt.title(f'{img_name} diagonal direction of the cipher image')
    plt.xlabel('Pixel gray value on location (x, y)')
    plt.ylabel('Pixel gray value on location (x+1, y+1)')
    plt.savefig(f'{path_diagonal}/{img_name}_cipher_image.png')
    plt.show()