import cv2 as cv
import numpy as np
import math
import random_function
import timeit
import copy

def join_column(arr1, arr2):

    arr12 = np.column_stack((arr1, arr2))

    return arr12

def join_row(arr1, arr2):

    arr12 = np.row_stack((arr1, arr2))

    return arr12