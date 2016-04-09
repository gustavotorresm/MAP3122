import numpy as np
import matplotlib.pyplot as plt

a = np.load('3.npy')

b = np.load('7.npy')

c = np.load('11.npy')

d = np.load('real.npy')

e = np.load('run.npy')

plt.plot(a[0], a[1], b[0], b[1], '--', c[0], c[1], ':', d[0], d[1], e[0], e[1])
plt.show()