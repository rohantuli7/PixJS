import numpy as np

def splitting(arr, split_no, h, w):

    arr = [arr[0:split_no, 0:split_no], arr[0:split_no, split_no:w],
           arr[split_no:h, 0:split_no], arr[split_no:h, split_no:w]]

    return arr