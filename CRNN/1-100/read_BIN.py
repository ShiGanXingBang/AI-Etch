import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl

path = 'test10.bin'
height = 500
# 1-100组height=500，101-200组height=700
width = 200

A = np.fromfile(path)
Si = np.reshape(A, (height, width))
colors = ['white', 'blue', 'orange', 'red', 'black']
bounds = [0, 1, 2, 3, 4]
cmap = mpl.colors.ListedColormap(colors)
norm = mpl.colors.BoundaryNorm(bounds, cmap.N)
plt.imshow(Si, interpolation='none', cmap=cmap, norm=norm)
plt.show()