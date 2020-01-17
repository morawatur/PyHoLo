import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm
from mpl_toolkits.mplot3d import Axes3D

import Dm3Reader as dm3

phs_file = 'G:\Holografia_Ogulnie\LNT\\usrednianie\\07_23_-10%__25_0%\diff_23_25\\res\\bins\phs3d\\diff_23_-10%_25_0%.dm3'

img_data, px_dims = dm3.ReadDm3File(phs_file)
phs = img_data.astype(np.float32)

# save 3d plot - phase(x, y)

img_dim = phs.shape[0]
px_sz = px_dims[0] * 1e9

# step = 50

X = np.arange(0, img_dim, dtype=np.float32)
Y = np.arange(0, img_dim, dtype=np.float32)
X, Y = np.meshgrid(X, Y)
X *= px_sz
Y *= px_sz

fig = plt.figure()
ax = fig.gca(projection='3d')
ax.plot_surface(X, Y, phs, cmap=cm.jet, rstride=50, cstride=50)
plt.show()
# plt.savefig('phase3d.png', dpi=300)