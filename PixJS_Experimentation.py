import cv2 as cv
import numpy as np
import math
import random_function
import timeit
import copy
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

# Taking Image Input and displaying it
def image_features(img_original, img_enc):
    # 1. Entropy
    entropy_og = shannon_entropy(img_original)
    entropy_enc = shannon_entropy(img_enc)

    # GLCM creation for the following features:
    # (i). Contrast
    # (ii). Correlation

    # Creation of GLCM
    glcm_og = greycomatrix(img_original, [1], [0, np.pi/4, np.pi/2])
    glcm_enc = greycomatrix(img_enc, [1], [0, np.pi/4, np.pi/2])

    # 2. Contrast
    contrast_og = greycoprops(glcm_og, 'contrast')
    contrast_enc = greycoprops(glcm_enc, 'contrast')


    # 3. Correlation
    correlation_og = greycoprops(glcm_og, 'correlation')
    correlation_enc = greycoprops(glcm_enc, 'correlation')

    # 4. Number of Pixel Change Rate (NPCR)
    #In temp pass encrypted image and original image
    temp = cv.absdiff(img_original, img_enc)
    temp = temp.astype(np.uint8)
    NPCR = (np.count_nonzero(temp) * 100) / temp.size

    # 5. Unified Average Changing Intensity
    temp = cv.absdiff(img_original, img_enc)/255
    UACI = (np.sum(temp) * 100) / temp.size

    occurences = [np.count_nonzero(img_enc == i) for i in range(0, 256)]
    e = int(img_enc.size/256)
    chi_test = [(np.square(i - e))/e for i in occurences]
    chi_test = np.sum(chi_test)

    return chi_test, entropy_og, entropy_enc, contrast_og, contrast_enc, correlation_og, correlation_enc, NPCR, UACI



def PIXJS(path, iterations, key_value):
    #im = cv.imread(path)
    im1 = path
    #im1 = cv.cvtColor(im, cv.COLOR_BGR2GRAY)
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

    # print('Tuli Super gandu', encrypted_image.shape)
    ##XOR - Chaos based

    seed = '10111101'
    h, w = encrypted_image.shape

    # STEPWISE IMPLEMENTATION OF CHAOS BASED LOGISTIC MAP
    P = chaos_based.convert_to_binary(encrypted_image)  # binary conversion

    # 1. GENERATING THE LOGISTIC MAP
    K1 = chaos_based.logistic_map(h, w, key_value)

    # 2. GENERATING THE LINEAR SHIFT REGISTER SEQUENCE
    K2 = chaos_based.linear_shift_register(seed, h, w)

    # 3. XOR OF K1 & K2
    # FINAL CHAOTIC SEQUENCE
    K3 = np.array([np.binary_repr(int(i, 2) ^ int(j, 2), width=8) for i, j in zip(K1.flatten(), K2.flatten())])

    # GENERATING THE ENCRYPTED IMAGE WITH CHAOS METHOD
    P_PRIME = np.array([np.binary_repr(int(i, 2) ^ int(j, 2), width=8) for i, j in zip(K3, P.flatten())])
    chaos_encrypted_image = np.array([int(i, 2) for i in P_PRIME]).reshape((h, w)).astype('uint8')  ##
    # print('Tuli Gandu', chaos_encrypted_image.shape)
    # cv.imshow('Encrypted Image Before', encrypted_image)
    # cv.waitKey(0)
    # cv.imshow('Encrypted Image', chaos_encrypted_image)
    # cv.waitKey(0)

    enc_stop = timeit.default_timer()

    dec_start = timeit.default_timer()
    # DECRYPTION PROCESS FROM CHAOS BASED
    STEP1 = chaos_based.convert_to_binary(chaos_encrypted_image)

    STEP2 = np.array([np.binary_repr(int(i, 2) ^ int(j, 2), width=8) for i, j in zip(K3, STEP1.flatten())])

    STEP3 = np.array([int(i, 2) for i in STEP2]).reshape((h, w))

    # unxor = cv.bitwise_xor(enc, xored)

    # comparison = STEP3 == encrypted_image
    # result = comparison.all()
    # print('eofkeopvjdpvnredwiovnrdi0vbrenb Result', result)

    # plt.hist(encrypted_image.ravel(), bins=256)
    # plt.show()
    #
    # plt.hist(chaos_encrypted_image.ravel(), bins=256)
    # plt.show()

    h, w = STEP3.shape
    # arr1234, trans_main1 = checking_h_w.checker(arr1234, h, w)
    if trans_main1 == 1:
        arr1234 = checking_h_w.flip_back(STEP3)
    else:
        arr1234 = STEP3

    if extra == 1:
        # print('aaaadsdsfsf')
        # Jumbling Salting again on the whole image
        h, w = arr1234.shape
        arrextra = arr1234.flatten()
        # print(rand1)
        for i in range(iterations):
            if i % 2 != 0:
                arrextra = np.flip(arrextra)
            arrextra = Encryption_Decryption.Decryption_flat(arrextra, rand1)

        arr1234 = np.reshape(arrextra, (h, w))

    f_ex = copy.deepcopy(arr1234)
    h, w = arr1234.shape

    # print(h, w)
    arr1234 = [arr1234[0:split_no, 0:split_no], arr1234[0:split_no, split_no:h],
               arr1234[split_no:h, 0:split_no], arr1234[split_no:h, split_no:h]]

    arr1 = arr1234[0]
    arr2 = arr1234[1]
    arr3 = arr1234[2]
    arr4 = arr1234[3]

    arr12 = joining_matrix.join_column(arr1, arr2)
    arr34 = joining_matrix.join_column(arr3, arr4)

    h12, w12 = arr12.shape
    h34, w34 = arr34.shape

    for i in range(iterations):
        arr12 = arr12.flatten()
        arr34 = arr34.flatten()

        # ha12 = len(arr12) / 2
        # ha12 = int(ha12)
        # rand1234 = random_function.random_generator(int(1.5 * ha12), len(arr12) - 1)

        for i in range(iterations):
            if i % 2 != 0:
                arr12 = np.flip(arr12)
                arr34 = np.flip(arr34)
            arr12 = Encryption_Decryption.Decryption_flat(arr12, rand1234)
            arr34 = Encryption_Decryption.Decryption_flat(arr34, rand1234)

        arr12 = np.reshape(arr12, (h12, w12))
        arr34 = np.reshape(arr34, (h34, w34))

    arre = [arr12[0:split_no, 0:split_no], arr12[0:split_no, split_no:h], arr34[0:split_no, 0:split_no],
            arr34[0:split_no, split_no:h]]

    arr1 = arre[0]
    arr2 = arre[1]
    arr3 = arre[2]
    arr4 = arre[3]


    # Getting New height and weight

    h1, w1 = arr1.shape
    h2, w2 = arr2.shape
    h3, w3 = arr3.shape
    h4, w4 = arr4.shape
    # print(w1, w2, w3, w4)

    if extra == 1:
        arrextra = f_ex[0:h, h:w]

        arrextra = arrextra.flatten()
        # print('')

    for i in range(iterations):

        if i % 2 != 0:
            # print('absbs')
            arr1 = np.transpose(arr1)
            arr2 = np.transpose(arr2)
            arr3 = np.transpose(arr3)
            arr4 = np.transpose(arr4)

        arr1 = Decryption.decrytion(arr1, h1, w1, mat_max=h1 - 1, iterator_row=h1 - 1, iterator_col=w1 - 1)
        arr2 = Decryption.decrytion(arr2, h2, w2, mat_max=h2 - 1, iterator_row=h2 - 1, iterator_col=w2 - 1)
        arr3 = Decryption.decrytion(arr3, h3, w3, mat_max=h3 - 1, iterator_row=h3 - 1, iterator_col=w3 - 1)
        arr4 = Decryption.decrytion(arr4, h4, w4, mat_max=h4 - 1, iterator_row=h4 - 1, iterator_col=w4 - 1)
        if extra == 1:
            # if i % 2 != 0:
            #     arrextra = np.flip(arrextra)
            arrextra = Encryption_Decryption.Decryption_flat(arrextra, rand)

    arr12 = joining_matrix.join_column(arr1, arr2)
    arr34 = joining_matrix.join_column(arr3, arr4)

    arr1234 = joining_matrix.join_row(arr12, arr34)

    if extra == 1:
        arrextra = np.reshape(arrextra, (h, w - h))

        # encrypted_image_part5 = np.array(arrextra).astype('uint8')
        #
        # cv.imshow('Encrypted Image Part5', encrypted_image_part5)
        # cv.waitKey(0)
        arr1234 = joining_matrix.join_column(arr1234, arrextra)
        h, w = arr1234.shape

    if trans_main1 == 1:
        arr1234 = checking_h_w.flip_back(arr1234)

    chi_test, entropy_og, entropy_enc, contrast_og, contrast_enc, correlation_og, correlation_enc, NPCR, UACI = image_features(orig, chaos_encrypted_image)

    return orig.shape, \
           chi_test, round(entropy_og, 6), round(entropy_enc, 6), \
           round(contrast_og[0][0], 6), round(contrast_enc[0][0], 6), \
           round(correlation_og[0][0], 6), round(correlation_enc[0][0], 6), round(correlation_og[0][1], 6), \
           round(correlation_enc[0][1], 6), round(correlation_og[0][2], 6), round(correlation_enc[0][2], 6), \
           round(NPCR, 6), round(UACI, 6), \
           orig, chaos_encrypted_image