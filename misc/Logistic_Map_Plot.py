import numpy as np
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings("ignore")

def logistic(r, x):
    return r * x * (1 - x)

n = 10000
r = np.linspace(2.5, 4.0, n)
iterations = 1000
last = 100
x = 1e-5 * np.ones(n)

for i in range(iterations):
    x = logistic(r, x)
    if i >= (iterations - last):
        plt.plot(r, x, ',k', alpha=.25, c = 'skyblue')
plt.title('Sequence generation using Logistic map')
plt.ylabel('$\it{x}$')
plt.xlabel('$\it{r}$')
#plt.savefig('/Users/rt/Desktop/PixJS/diagrams/Logistic_Map.png')
plt.show()