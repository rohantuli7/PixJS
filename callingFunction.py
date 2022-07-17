import PixJS_Main
import pandas as pd
import numpy as np
import os
import cv2 as cv
import matplotlib.pyplot as plt
from matplotlib import colors
from matplotlib.ticker import PercentFormatter
import RQ4_random

cols = ['IMG_NAME', 'SIZE',
        'ENCRYPTION TIME', 'DECRYPTION TIME',
        'CHI TEST', 'ENTROPY BEFORE ENCRYPTION', 'ENTROPY AFTER ENCRYPTION',
        'CONTRAST BEFORE ENCRYPTION', 'CONTRAST AFTER ENCRYPTION ',
        'HORIZONTAL CORRELATION BEFORE ENCRYPTION', 'HORIZONTAL CORRELATION AFTER ENCRYPTION',
        'DIAGONAL CORRELATION BEFORE ENCRYPTION', 'DIAGONAL CORRELATION AFTER ENCRYPTION',
        'VERTICAL CORRELATION BEFORE ENCRYPTION', 'VERTICAL CORRELATION AFTER ENCRYPTION',
        'NPCR', 'UACI']
df = pd.DataFrame(columns = cols)
images = {}
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
    df_temp = pd.DataFrame(columns=cols)
    for i, j in zip(img_path, img_name):
        image_features = list(PixJS_Main.PIXJS(i, 1))
        file_name = j.replace('.jpeg', '')
        image_features.insert(0, file_name)
        x = image_features.pop()
        cipher_img = x
        #plt.scatter(x)
        #plt.show()
        #cv.imwrite(f'/Users/rt/Desktop/PixJS/readings/Encrypted image/{file_name}_BEFORE_ENCRYPTION.png', x)
        # plt.hist(x.astype(np.uint8).ravel(), bins=256, color = "darkblue")
        # plt.title(f'{file_name} AFTER ENCRYPTION')
        # plt.savefig(f'/Users/rt/Desktop/PixJS/readings/histograms/after/{file_name}_AFTER_ENCRYPTION.png')
        # plt.show()
        #
        #plt.scatter(x)
        #plt.show()
        x = image_features.pop()
        plain_img = x

        RQ4_random.correlation_analysis(plain_img, cipher_img, file_name)
        #plt.scatter(x)
        #plt.show()
        #cv.imwrite(f'/Users/rt/Desktop/PixJS/readings/Original image/{file_name}_AFTER_ENCRYPTION.png', x)
        # plt.hist(x.astype(np.uint8).ravel(), bins=256, color = "skyblue")
        # plt.title(f'{file_name} BEFORE ENCRYPTION')
        # plt.savefig(f'/Users/rt/Desktop/PixJS/readings/histograms/before/{file_name}_BEFORE_ENCRYPTION.png')
        # plt.show()
        mydict = {k: v for (k, v) in zip(cols, image_features)}
        df_temp = df_temp.append(mydict, ignore_index=True)
    return df_temp

path_sym_small = '/Users/rt/Desktop/College/projects/PixJS/FINAL DATABASE/SYM SMALL'
path_sym_big = '/Users/rt/Desktop/College/projects/PixJS/FINAL DATABASE/SYM BIG'
path_asym_small = '/Users/rt/Desktop/College/projects/PixJS/FINAL DATABASE/ASYM SMALL'
path_asym_big = '/Users/rt/Desktop/College/projects/PixJS/FINAL DATABASE/ASYM BIG'

df = df.append(image_features_function(path_sym_small), ignore_index=True)
df = df.append(image_features_function(path_sym_big), ignore_index=True)
df = df.append(image_features_function(path_asym_small), ignore_index=True)
df = df.append(image_features_function(path_asym_big), ignore_index=True)

df.to_csv('/Users/rt/Desktop/PIXJS_Metrics.csv')


# from skimage.feature import greycomatrix
# from skimage.feature import greycoprops
#
# glcm_og = greycomatrix(img, [1], [0, np.pi/4, np.pi/2])
# contrast_og = greycoprops(glcm_og, 'contrast')
# correlation_og = greycoprops(glcm_og, 'correlation')
#
#path = '/Users/rt/Desktop/IMG_SYM_BIG_6.jpeg'
#
#img = cv.imread('/Users/rt/Desktop/IMG_SYM_BIG_6.jpeg')
#img = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
#
#print(list(PixJS_copy.PIXJS(path, 1)))
#
# x = np.where(img == img, 0, 1)
# print(np.count_nonzero(img))
#plt.imshow(img)
#plt.title('nejnfjenejff')
#plt.savefig('/Users/rt/Desktop/ejfenf.png')
# Remove x, y ticks
# Creating histogram

# occurences = [np.count_nonzero(img == i) for i in range(0, 256)]
# e = int(img.size/256)
# hello = [np.square((i - e)/e) for i in occurences]
# hello = np.sum(hello)