import cv2 as cv
import numpy as np
import math
import random_function
import timeit
import copy

def jumbling(arr, h, w, mat_max, iterator_row, iterator_col):
    while(mat_max!=h):
        l = []
        x = []

        for i in range(w): #Taking L ka value
            l.append(arr[iterator_row][i])

        for i in range(h): # Taking x ka value
            if i == iterator_col:
                pass
            else:
                if i == w:
                    pass
                else:
                    x.append(arr[i][iterator_col])

        xl = x + l

        # Doing Jumbling

        j = len(l)
        i = 0
        fix = len(xl)
        start_time = timeit.default_timer()
        while (j != 0):
            temp = 0
            temp = fix % j
            xl[i], xl[temp] = xl[temp], xl[i]
            j = j - 1
            i = i + 1

        # print(xl)

        #Replacing values back in the matrix

        x = xl[:len(x)]
        l = xl[len(x):]

        for i in range(w): #Taking L ka value
            arr[iterator_row][i] = l[i]

        j = 0
        for i in range(h): # Taking x ka value
            if i == iterator_col:
                pass
            else:
                if i == w:
                    pass
                else:
                    arr[i][iterator_col] = x[j]
                    j = j + 1


        iterator_row = iterator_row + 1
        iterator_col = iterator_col + 1
        mat_max = mat_max + 1

    return arr