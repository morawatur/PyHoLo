# matplotlib.use('agg')
import matplotlib.pyplot as pplt
import numpy as np

import Dm3Reader as dm3
import ImageSupport as imsup

# ---------------------------------------------------------

# def func_to_vectorize(x, y, dx, dy, scaling=1):
#     pplt.arrow(x, y, dx*scaling, dy*scaling, fc="k", ec="k", head_width=0.06, head_length=0.1)
#
# vectorized_arrow_drawing = np.vectorize(func_to_vectorize)

# ---------------------------------------------------------

# def draw_image_with_gradient_arrows(arr, step=20):
img_path = 'test1.dm3'
imgData, pxDims = dm3.ReadDm3File(img_path)
imsup.Image.px_dim_default = pxDims[0]
img = imsup.ImageExp(imgData.shape[0], imgData.shape[1], imsup.Image.cmp['CAP'], num=1, px_dim_sz=pxDims[0])
img.LoadAmpData(imgData.astype(np.float32))
img.amPh.ph = np.copy(img.amPh.am)  # !!!

fig = pplt.figure()
ax = fig.add_subplot(111)
width, height = img.amPh.ph.shape
step = 60
offset = 50
h_min = offset
h_max = width - offset
h_dist = h_max - h_min
step = h_dist / float(np.ceil(h_dist / step))
xv, yv = np.meshgrid(np.arange(0, h_dist, step), np.arange(0, h_dist, step))
xv += step / 2.0
yv += step / 2.0
print(step)

yd, xd = np.gradient(img.amPh.ph)
ydd = yd[h_min:h_max:60, h_min:h_max:60]
xdd = xd[h_min:h_max:60, h_min:h_max:60]

avg_ydd = np.average(np.abs(ydd))
avg_xdd = np.average(np.abs(xdd))

print(avg_xdd, avg_ydd)
print(ydd)

ydd[np.abs(ydd) > 5 * avg_ydd] = 0.0
for x in range(ydd.shape[0]):
    for y in range(ydd.shape[1]):
        if ydd[x, y] == 0.0:
            ydd_ngb = ydd[x-1:x+2, y-1:y+2]
            ydd[x, y] = np.sum(ydd_ngb) / (ydd_ngb.size-1)

xdd[np.abs(xdd) > 5 * avg_xdd] = 0.0
for x in range(xdd.shape[0]):
    for y in range(xdd.shape[1]):
        if xdd[x, y] == 0.0:
            xdd_ngb = xdd[x-1:x+2, y-1:y+2]
            xdd[x, y] = np.sum(xdd_ngb) / (xdd_ngb.size-1)

print(xv.shape)
print(xdd.shape)

def func_to_vectorize(x, y, dx, dy, sc=1):
    pplt.arrow(x, y, dx*sc, dy*sc, fc="yellow", ec="k", lw=0.6, head_width=14, head_length=18)

vectorized_arrow_drawing = np.vectorize(func_to_vectorize)

ph_roi = np.copy(img.amPh.ph[h_min:h_max, h_min:h_max])
# amp_factor = 4
# ph_roi = np.cos(amp_factor * ph_roi)

dr_d = np.hypot(xd, yd)
ph_d = np.arctan2(xd, yd)

dr_d_roi = np.copy(dr_d[h_min:h_max, h_min:h_max])
ph_d_roi = np.copy(ph_d[h_min:h_max, h_min:h_max])

pplt.imshow(ph_roi)
# ax.imshow(arr)
vectorized_arrow_drawing(xv, yv, xdd, ydd, 3000)
# pplt.set_cmap(pplt.cm.Greys)
pplt.set_cmap(pplt.cm.RdYlBu)
pplt.colorbar()
# fig.savefig('demo.png')     # !!!
pplt.show()