import numpy as np

def checker(arr, h, w):
    trans_main = 0
    if h > w:
        arr = np.transpose(arr)
        trans_main = trans_main + 1

    return arr, trans_main

def flip_back(arr):

    arr = np.transpose(arr)

    return arr