import cv2 as cv
import numpy as np
import math
import random_function
import timeit
import copy as cp
import checking_h_w
import Jumbling
import joining_matrix
import Decryption
import Encryption_Decryption
import matplotlib.pyplot as plt
from scipy.stats import chi2
import os
from tabulate import tabulate
import chaos_based
from prettytable import PrettyTable
from skimage.feature import greycomatrix
from skimage.feature import greycoprops
from skimage.measure import shannon_entropy
from numpy.random import rand
import random
import copy

def calc_UACI(enc_img, orig_img):
    temp = cv.absdiff(orig_img, enc_img)/255
    return (np.sum(temp) * 100) / temp.size

def calc_NPCR(enc_img, orig_img):
    temp = cv.absdiff(orig_img, enc_img)
    temp = temp.astype(np.uint8)
    return (np.count_nonzero(temp) * 100) / temp.size

path = "/Users/rt/Desktop/College/Misc/projects/PixJS/FINAL DATABASE/SYM SMALL/IMG_SYM_SMALL_1.jpeg"

# path = '/Users/rt/Desktop/College/projects/PixJS/FINAL DATABASE/SYM SMALL/IMG_SYM_SMALL_1.jpeg'

def encrypt(img):
    iterations = 1
    im1 = img
    original_img = im1
    # Getting Image height and width and splitting it
    enc_start = timeit.default_timer()
    arr = np.asarray(im1)
    h, w = arr.shape
    orig = copy.deepcopy(arr)

    extra = 0
    if h != w:
        extra = 1

    arr, trans_main1 = checking_h_w.checker(arr, h, w)
    h, w = arr.shape

    f_ex = copy.deepcopy(arr)
    half = h / 2

    # split_no = random_function.random_generator(1, w)
    split_no = int(half)

    # print(split_no)
    # print(arr.shape)
    arr = [arr[0:split_no, 0:split_no], arr[0:split_no, split_no:h],
           arr[split_no:h, 0:split_no], arr[split_no:h, split_no:h]]

    # Getting height and width of each part and then checking for flip

    arr1 = arr[0]
    arr2 = arr[1]
    arr3 = arr[2]
    arr4 = arr[3]

    # Getting New height and weight

    h1, w1 = arr1.shape
    h2, w2 = arr2.shape
    h3, w3 = arr3.shape
    h4, w4 = arr4.shape
    # print(w1, w2, w3, w4)
    # print(h1, h2, h3, h4)
    arr12 = joining_matrix.join_column(arr1, arr2)
    arr34 = joining_matrix.join_column(arr3, arr4)

    # Extra Image part
    if extra == 1:
        arrextra = f_ex[0:h, h:w]

        arrextra = arrextra.flatten()
        ha = len(arrextra) / 2
        ha = int(ha)
        rand = random_function.random_generator(ha, (len(arrextra) - 1))

    # while (True):
    #     iterations = random_function.random_generator(10, 20)
    #     if iterations % 2 == 0:
    #         break
    #     else:
    #         continue

    #iterations = 1

    for i in range(iterations):
        if i % 2 != 0:
            arr1 = np.transpose(arr1)
            arr2 = np.transpose(arr2)
            arr3 = np.transpose(arr3)
            arr4 = np.transpose(arr4)

        arr1 = Jumbling.jumbling(arr1, h1, w1, mat_max=0, iterator_col=0, iterator_row=0)
        arr2 = Jumbling.jumbling(arr2, h2, w2, mat_max=0, iterator_col=0, iterator_row=0)
        arr3 = Jumbling.jumbling(arr3, h3, w3, mat_max=0, iterator_col=0, iterator_row=0)
        arr4 = Jumbling.jumbling(arr4, h4, w4, mat_max=0, iterator_col=0, iterator_row=0)

        if extra == 1:
            # if i % 2 != 0:
            # arrextra = np.flip(arrextra)
            arrextra = Encryption_Decryption.Jumbling_flat(arrextra, rand)

    # encrypted_image_part1 = np.array(arr1).astype('uint8')
    # cv.imshow('Encrypted Image Part1', encrypted_image_part1)
    # cv.waitKey(0)
    #
    # encrypted_image_part2 = np.array(arr2).astype('uint8')
    # cv.imshow('Encrypted Image Part2', encrypted_image_part2)
    # cv.waitKey(0)
    #
    # encrypted_image_part3 = np.array(arr3).astype('uint8')
    # cv.imshow('Encrypted Image Part3', encrypted_image_part3)
    # cv.waitKey(0)
    #
    # encrypted_image_part4 = np.array(arr4).astype('uint8')
    # cv.imshow('Encrypted Image Part4', encrypted_image_part4)
    # cv.waitKey(0)

    arr12 = joining_matrix.join_column(arr1, arr2)
    arr34 = joining_matrix.join_column(arr3, arr4)

    h12, w12 = arr12.shape
    h34, w34 = arr34.shape

    for i in range(iterations):
        arr12 = arr12.flatten()
        arr34 = arr34.flatten()

        ha12 = len(arr12) / 2
        ha12 = int(ha12)
        rand1234 = random_function.random_generator(int(1.5*ha12), len(arr12) - 1)

        for i in range(iterations):
            if i % 2 != 0:
                arr12 = np.flip(arr12)
                arr34 = np.flip(arr34)
            arr12 = Encryption_Decryption.Jumbling_flat(arr12, rand1234)
            arr34 = Encryption_Decryption.Jumbling_flat(arr34, rand1234)

        arr12 = np.reshape(arr12, (h12, w12))
        arr34 = np.reshape(arr34, (h34, w34))


    arr1234 = joining_matrix.join_row(arr12, arr34)

    if extra == 1:
        h, w = f_ex.shape
        arrextra = np.reshape(arrextra, (h, w - h))

        # encrypted_image_part5 = np.array(arrextra).astype('uint8')

        arr1234 = joining_matrix.join_column(arr1234, arrextra)
        h, w = arr1234.shape
        # Jumbling Salting again on the whole image

        arrextra = arr1234.flatten()
        ha1 = len(arrextra) / 2
        ha1 = int(ha1)
        rand1 = random_function.random_generator(int(1.5 * ha1), (len(arrextra) - 1))
        # print(rand1)
        for i in range(iterations):
            if i % 2 != 0:
                arrextra = np.flip(arrextra)
            arrextra = Encryption_Decryption.Jumbling_flat(arrextra, rand1)

        arr1234 = np.reshape(arrextra, (h, w))

    if trans_main1 == 1:
        arr1234 = checking_h_w.flip_back(arr1234)

    # print(arr1234.shape)
    encrypted_image = np.array(arr1234).astype('uint8')

    ##XOR - Chaos based

    seed = '10111101'
    h, w = encrypted_image.shape

    # STEPWISE IMPLEMENTATION OF CHAOS BASED LOGISTIC MAP
    P = chaos_based.convert_to_binary(encrypted_image)  # binary conversion

    # 1. GENERATING THE LOGISTIC MAP
    K1 = chaos_based.logistic_map(h, w, 0.1)

    # 2. GENERATING THE LINEAR SHIFT REGISTER SEQUENCE
    K2 = chaos_based.linear_shift_register(seed, h, w)

    # 3. XOR OF K1 & K2
    # FINAL CHAOTIC SEQUENCE
    K3 = np.array([np.binary_repr(int(i, 2) ^ int(j, 2), width=8) for i, j in zip(K1.flatten(), K2.flatten())])

    # GENERATING THE ENCRYPTED IMAGE WITH CHAOS METHOD
    P_PRIME = np.array([np.binary_repr(int(i, 2) ^ int(j, 2), width=8) for i, j in zip(K3, P.flatten())])
    chaos_encrypted_image = np.array([int(i, 2) for i in P_PRIME]).reshape((h, w)).astype('uint8')
    return chaos_encrypted_image

img = cv.imread(path, 0)
height, width = img.shape
change_x = np.random.randint(0, width)
change_y = np.random.randint(0, height)
val = np.random.randint(0, 255)

new_img = cp.deepcopy(img)
new_img[change_x, change_y] = val
print(np.array_equal(img, new_img))

plt.imshow(img)
plt.show()

plt.imshow(new_img)
plt.show()

ENC1 = encrypt(img)
ENC2 = encrypt(new_img)
print(calc_UACI(ENC1, ENC2))

plt.imshow(ENC1)
plt.show()

plt.imshow(ENC2)
plt.show()