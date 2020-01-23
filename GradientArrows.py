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

import matplotlib.pyplot as plt
import numpy as np

import Dm3Reader as dm3
import ImageSupport as imsup

from os import listdir
from os.path import isfile, join

# ---------------------------------------------------------

def func_to_vectorize(x, y, dx, dy, sc=1):
    plt.arrow(x, y, dx*sc, dy*sc, fc="k", ec="k", lw=0.6, head_width=10, head_length=14)

# ---------------------------------------------------------

# def draw_image_with_gradient_arrows(arr, step=20):

# img_dir = 'input/strzalki'
# img_dir = 'input/FeSiB_brzeg'
# img_dir = 'input/temp'
img_dir = 'G:\Holografia_Ogulnie\LNT\profile_kolaz\\nowy_slimak2\cz1\\bins\glob_scale'
files = [ '{0}/{1}'.format(img_dir, f) for f in listdir(img_dir) if isfile(join(img_dir, f)) and f.endswith('.dm3') ]
global_limits = [1e5, 0]

for f in files:
    img_data, px_dims = dm3.ReadDm3File(f)
    phs = img_data.astype(np.float32)
    limits = [np.min(phs), np.max(phs)]
    if limits[0] < global_limits[0]:
        global_limits[0] = limits[0]
    if limits[1] > global_limits[1]:
        global_limits[1] = limits[1]

# ---------------------------------------------------------

idx = 1

for f in files:
    print(f)
    img_data, px_dims = dm3.ReadDm3File(f)
    imsup.Image.px_dim_default = px_dims[0]
    img = imsup.ImageExp(img_data.shape[0], img_data.shape[1], imsup.Image.cmp['CAP'], num=1, px_dim_sz=px_dims[0])
    img.LoadAmpData(img_data.astype(np.float32))
    img.amPh.ph = np.copy(img.amPh.am)

    fig = plt.figure()
    width, height = img.amPh.ph.shape
    def_step = 50
    offset = 0
    h_min = offset
    h_max = width - offset
    h_dist = h_max - h_min
    step = h_dist / float(np.ceil(h_dist / def_step))
    xv, yv = np.meshgrid(np.arange(0, h_dist, step), np.arange(0, h_dist, step))
    xv += step / 2.0
    yv += step / 2.0
    # print(step)

    yd, xd = np.gradient(img.amPh.ph)
    ydd = yd[h_min:h_max:def_step, h_min:h_max:def_step]
    xdd = xd[h_min:h_max:def_step, h_min:h_max:def_step]

    # prevent artifacts
    # avg_ydd = np.average(np.abs(ydd))
    # avg_xdd = np.average(np.abs(xdd))
    #
    # # print(avg_xdd, avg_ydd)
    # # print(ydd)
    #
    # ydd[np.abs(ydd) > 5 * avg_ydd] = 0.0
    # for x in range(ydd.shape[0]):
    #     for y in range(ydd.shape[1]):
    #         if ydd[x, y] == 0.0:
    #             ydd_ngb = ydd[x-1:x+2, y-1:y+2]
    #             ydd[x, y] = np.sum(ydd_ngb) / (ydd_ngb.size-1)
    #
    # xdd[np.abs(xdd) > 5 * avg_xdd] = 0.0
    # for x in range(xdd.shape[0]):
    #     for y in range(xdd.shape[1]):
    #         if xdd[x, y] == 0.0:
    #             xdd_ngb = xdd[x-1:x+2, y-1:y+2]
    #             xdd[x, y] = np.sum(xdd_ngb) / (xdd_ngb.size-1)

    # print(xv.shape)
    # print(xdd.shape)

    vectorized_arrow_drawing = np.vectorize(func_to_vectorize)

    ph_roi = np.copy(img.amPh.ph[h_min:h_max, h_min:h_max])
    # amp_factor = 4
    # ph_roi = np.cos(amp_factor * ph_roi)

    # dr_d = np.hypot(xd, yd)
    # ph_d = np.arctan2(xd, yd)

    # dr_d_roi = np.copy(dr_d[h_min:h_max, h_min:h_max])
    # ph_d_roi = np.copy(ph_d[h_min:h_max, h_min:h_max])

    plt.imshow(ph_roi, vmin=global_limits[0], vmax=global_limits[1], cmap=plt.cm.get_cmap('jet'))
    # vectorized_arrow_drawing(xv, yv, xdd, ydd, 4000)      # pokazuje / ukrywa strzalki
    # plt.set_cmap(pplt.cm.Greys)
    # plt.set_cmap('jet')
    plt.axis('off')

    if idx == len(files):
        cbar = plt.colorbar(label='phase shift [rad]')
        cbar.set_label('phase shift [rad]') # rotation=270)

    idx += 1
    out_f = f.replace('dm3', 'png')
    plt.savefig(out_f, dpi=300)
    # plt.savefig(out_f, dpi=300, bbox_inches = 'tight')
    # plt.show()