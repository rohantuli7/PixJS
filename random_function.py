# import random
# import numpy as np
# def random_generator(min, max):
#     r = random.randint(min+1, max-1)
#
#     if r%2 == 0:
#         x = np.random.randint(r, max)
#     else:
#         x = np.random.randint(min, r)
#
#     return x



import numpy as np
import random

def random_generator(min, max):
    num = np.random.poisson(5, 1)
    if num == 0:
        num = random.randint(min+1, max-1)
        return num
    else:
        return num[0]

#print(random_generator(5, 15))
