import numpy as np

def convert_to_binary(img):
    lst = []
    for i in range(img.shape[0]):
        for j in range(img.shape[1]):
            lst.append(np.binary_repr(img[i][j] ,width=8))
    lst = np.array(lst)
    return lst.reshape((img.shape[0], img.shape[1]))

def logistic_map(height, width, key_value):
    size = height * width
    r = 3.99
    logistic_map = []
    x0 = key_value
    for i in range(size):
        temp = r * x0 * (1 - x0)
        x0 = temp
        logistic_map.append(temp)
    logistic_map = np.array(logistic_map)
    logistic_image = (logistic_map*255).astype('uint8')
    bin_img = [np.binary_repr(i, width = 8) for i in logistic_image]
    return np.array(bin_img)

def linear_shift_register(seed, height, width):
    size = height * width
    seed = seed
    lfsr = []
    lfsr.append(seed)
    for i in range(size-1):
        d0, d4, d5, d6 = seed[0], seed[4], seed[5], seed[6]
        xor = str(int(d0) ^ int(d4) ^ int(d5) ^ int(d6))
        seed = seed[1] + seed[2] + seed[3] + seed[4] + seed[5] + seed[6] + seed[7] + xor
        lfsr.append(seed)
    lfsr = np.array(lfsr).reshape((width, height))
    return lfsr

#CONVERTING ORIGINAL IMAGE TO BINARY

#STEPWISE IMPLEMENTATION OF CHAOS BASED LOGISITIC MAP
#1. GENERATING THE LOGISTIC MAP

#2. GENERATING THE LINEAR SHIFT REGISTER SEQUENCE

#3. XOR OF K1 & K2
#FINAL CHAOTIC SEQUENCE

#GENERATING THE ENCRYPTED IMAGE
# P_PRIME = np.array([np.binary_repr(int(i, 2) ^ int(j, 2), width = 8) for i, j in zip(K3, P.flatten())])
# encrypted_image = np.array([int(i, 2) for i in P_PRIME]).reshape((width, height))
#
# #DECRYPTION PROCESS
# STEP1 = convert_to_binary(encrypted_image)
#
# STEP2 = np.array([np.binary_repr(int(i, 2) ^ int(j, 2), width = 8) for i, j in zip(K3, STEP1.flatten())])
#
# STEP3 = np.array([int(i, 2) for i in STEP2]).reshape((width, height))