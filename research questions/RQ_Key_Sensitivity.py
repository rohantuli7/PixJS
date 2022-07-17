import PixJS_Experimentation
import os
import numpy as np
import matplotlib.pyplot as plt
import cv2 as cv

path_save = '/Users/rt/Desktop/PixJS/readings/RQ4'
'''
def image_features_function(basepath):
    img_path = []
    img_name = []
    for i in os.listdir(basepath):
        if i == '.DS_Store':
            pass
        else:
            img_path.append(os.path.join(basepath, i))
            img_name.append(i)
    img_name = sorted(img_name)
    img_path = sorted(img_path)
    for i, j in zip(img_path, img_name):
        K1 = PixJS_copy_copy.PIXJS(i, 1, 0.1)
        #cv.imwrite(f'{path_save}/{j}_cipher_1.jpeg', K1)
        plt.imshow(K1)
        plt.show()
        K2 = PixJS_copy_copy.PIXJS(i, 1, 0.1000000000000001)
        #cv.imwrite(f'{path_save}/{j}_cipher_2.jpeg', K2)
        plt.imshow(K2)
        plt.show()
        res = cv.absdiff(K1, K2)
        res = res.astype(np.uint8)
        percentage_difference = (np.count_nonzero(res) * 100)/res.size
        print(round(percentage_difference, 6))

path_sym_small = '/Users/rt/Desktop/PixJS/FINAL DATABASE/SYM SMALL'
path_sym_big = '/Users/rt/Desktop/PixJS/FINAL DATABASE/SYM BIG'
path_asym_small = '/Users/rt/Desktop/PixJS/FINAL DATABASE/ASYM SMALL'
path_asym_big = '/Users/rt/Desktop/PixJS/FINAL DATABASE/ASYM BIG'

image_features_function(path_sym_small)
image_features_function(path_sym_big)
image_features_function(path_asym_small)
image_features_function(path_asym_big)
'''

img_white = np.ones((512, 512), dtype=np.uint8) * 255
plt.hist(img_white.astype(np.uint8).ravel(), bins=256, color = "skyblue")
plt.savefig('/Users/rt/Desktop/PixJS/readings/RQ4 cryptanalysis/white_plain_hist.jpeg')
plt.show()

img_black = np.ones((512, 512), dtype=np.uint8)
plt.hist(img_black.astype(np.uint8).ravel(), bins=256, color = "skyblue")
plt.savefig('/Users/rt/Desktop/PixJS/readings/RQ4 cryptanalysis/black_plain_hist.jpeg')
plt.show()

cv.imwrite('/Users/rt/Desktop/PixJS/readings/RQ4 cryptanalysis/white_plain.jpeg', img_white)
cv.imwrite('/Users/rt/Desktop/PixJS/readings/RQ4 cryptanalysis/black_plain.jpeg', img_black)

cols = ['SIZE',
        'CHI TEST', 'ENTROPY BEFORE ENCRYPTION', 'ENTROPY AFTER ENCRYPTION',
        'CONTRAST BEFORE ENCRYPTION', 'CONTRAST AFTER ENCRYPTION ',
        'HORIZONTAL CORRELATION BEFORE ENCRYPTION', 'HORIZONTAL CORRELATION AFTER ENCRYPTION',
        'DIAGONAL CORRELATION BEFORE ENCRYPTION', 'DIAGONAL CORRELATION AFTER ENCRYPTION',
        'VERTICAL CORRELATION BEFORE ENCRYPTION', 'VERTICAL CORRELATION AFTER ENCRYPTION',
        'NPCR', 'UACI']

img_features_white = list(PixJS_Experimentation.PIXJS(img_white, 1, 0.1))
white_enc = img_features_white.pop()
cv.imwrite('/Users/rt/Desktop/PixJS/readings/RQ4 cryptanalysis/white_cipher.jpeg', white_enc)
plt.hist(white_enc.astype(np.uint8).ravel(), bins=256, color = "darkblue")
plt.savefig('/Users/rt/Desktop/PixJS/readings/RQ4 cryptanalysis/white_cipher_hist.jpeg')
plt.show()

mydict = {k: v for (k, v) in zip(cols, img_features_white)}
print('WHITE IMAGE FEATURES : ')
print(mydict)

img_features_black = list(PixJS_Experimentation.PIXJS(img_black, 1, 0.1))
black_enc = img_features_black.pop()
cv.imwrite('/Users/rt/Desktop/PixJS/readings/RQ4 cryptanalysis/black_cipher.jpeg', black_enc)
plt.hist(black_enc.astype(np.uint8).ravel(), bins=256, color = "darkblue")
plt.savefig('/Users/rt/Desktop/PixJS/readings/RQ4 cryptanalysis/black_cipher_hist.jpeg')
plt.show()

mydict = {k: v for (k, v) in zip(cols, img_features_black)}
print('BLACK IMAGE FEATURES : ')
print(mydict)
