# Copyright (C) 2020  Krzysztof Morawiec
#
# This file is part of PyHoLo.
#
# PyHoLo is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# PyHoLo is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.
#
# You should have received a copy of the GNU General Public License
# along with PyHoLo.  If not, see <https://www.gnu.org/licenses/>.

#-------------------------------------------------------------------

import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm
# from mpl_toolkits.mplot3d import Axes3D

import Dm3Reader as dm3

phs_file = 'C:\\example\\path\\phase_image.dm3'

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