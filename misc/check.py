import random
import cv2 as cv
import numpy as np
from numpy.random import rand

def calc_UACI(enc_img, orig_img):
    temp = cv.absdiff(orig_img, enc_img)/255
    return (np.sum(temp) * 100) / temp.size

def calc_NPCR(enc_img, orig_img):
    temp = cv.absdiff(orig_img, enc_img)
    temp = temp.astype(np.uint8)
    return (np.count_nonzero(temp) * 100) / temp.size

orig = np.uint8(rand(256, 256)*100)
enc = np.uint8(rand(256, 256)*100)
NPCRs = []
UACIs = []
for i in range(100):
    rand_pix = np.random.randint(256, size = (2,1))
    binary_value = np.binary_repr(orig[rand_pix[0][0]][rand_pix[1][0]], width = 8)
    random_bit = random.randint(0, 7)
    val = binary_value[random_bit]
    binary_value = list(binary_value)
    if val == '0':
        binary_value[random_bit] = '1'
    else:
        binary_value[random_bit] = '0'
    new_pixel_val = int("".join(binary_value), 2)
    orig[rand_pix[0][0]][rand_pix[1][0]] = new_pixel_val
    NPCRs.append(calc_NPCR(enc, orig))
    UACIs.append(calc_UACI(enc, orig))

print(np.mean(NPCRs))
print(np.mean(UACIs))