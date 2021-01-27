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
import scipy as sp
from PIL import Image as im

#-------------------------------------------------------------------

class ComplexAmPhMatrix:
    def __init__(self, height, width):
        self.am = np.zeros((height, width), dtype=np.float32)
        self.ph = np.zeros((height, width), dtype=np.float32)

    def __del__(self):
        del self.am
        del self.ph

#-------------------------------------------------------------------

def conj_am_ph_matrix(ap):
    ap_conj = ComplexAmPhMatrix(ap.am.shape[0], ap.am.shape[1])
    ap_conj.am = np.copy(ap.am)
    ap_conj.ph = -ap.ph
    return ap_conj

# -------------------------------------------------------------------

def mult_am_ph_matrices(ap1, ap2):
    ap3 = ComplexAmPhMatrix(ap1.am.shape[0], ap1.am.shape[1])
    ap3.am = ap1.am * ap2.am
    ap3.ph = ap1.ph + ap2.ph
    return ap3

#-------------------------------------------------------------------

class Image:
    cmp = {'CRI': 0, 'CAP': 1}
    capVar = {'AM': 0, 'PH': 1}
    criVar = {'RE': 0, 'IM': 1}
    px_dim_default = 1.0

    def __init__(self, height, width, cmpRepr=cmp['CAP'], defocus=0.0, num=1, px_dim_sz=-1.0):
        self.width = width
        self.height = height
        self.size = width * height
        self.name = ''
        self.reIm = np.zeros((height, width), dtype=np.complex64)
        self.amPh = ComplexAmPhMatrix(height, width)
        self.cmpRepr = cmpRepr
        self.defocus = defocus
        self.numInSeries = num
        self.prev = None
        self.next = None
        self.px_dim = px_dim_sz
        # self.bias = 0.0
        # self.gain = 255.0
        # self.gamma = 1.0
        if px_dim_sz < 0:
            self.px_dim = self.px_dim_default

    def __del__(self):
        self.reIm = None
        self.amPh.am = None
        self.amPh.ph = None

    def ChangeComplexRepr(self, newRepr):
        if newRepr == self.cmp['CAP']:
            self.ReIm2AmPh()
        elif newRepr == self.cmp['CRI']:
            self.AmPh2ReIm()

    def ReIm2AmPh(self):
        if self.cmpRepr == self.cmp['CAP']:
            return
        self.amPh.am = np.abs(self.reIm)
        self.amPh.ph = np.angle(self.reIm)
        # self.amPh.ph = np.arctan2(self.reIm.imag, self.reIm.real)
        self.cmpRepr = self.cmp['CAP']

    def AmPh2ReIm(self):
        if self.cmpRepr == self.cmp['CRI']:
            return
        self.reIm = self.amPh.am * np.exp(1j * self.amPh.ph)
        self.cmpRepr = self.cmp['CRI']

    def get_num_in_series_from_prev(self):
        if self.prev is not None:
            self.numInSeries = self.prev.numInSeries + 1

# -------------------------------------------------------------------

class ImageExp(Image):
    def __init__(self, height, width, cmpRepr=Image.cmp['CAP'], defocus=0.0, num=1, px_dim_sz=-1.0):
        super(ImageExp, self).__init__(height, width, cmpRepr, defocus, num, px_dim_sz)
        self.parent = super(ImageExp, self)
        self.shift = [0, 0]
        self.rot = 0
        self.amp_factor = 1.0
        self.cos_phase = None
        self.buffer = ComplexAmPhMatrix(height, width)

    def __del__(self):
        super(ImageExp, self).__del__()
        self.buffer.am = None
        self.buffer.ph = None
        self.cos_phase = None

    def LoadAmpData(self, ampData):
        self.amPh.am = np.copy(ampData)
        self.buffer.am = np.copy(ampData)

    def LoadPhsData(self, phsData):
        self.amPh.ph = np.copy(phsData)
        self.buffer.ph = np.copy(phsData)

    def UpdateBuffer(self):
        self.buffer.am = np.copy(self.amPh.am)
        self.buffer.ph = np.copy(self.amPh.ph)

    def UpdateImageFromBuffer(self):
        self.amPh.am = np.copy(self.buffer.am)
        self.amPh.ph = np.copy(self.buffer.ph)

    def ReIm2AmPh(self):
        super(ImageExp, self).ReIm2AmPh()
        self.UpdateBuffer()

    def update_cos_phase(self):
        self.cos_phase = np.cos(self.amPh.ph)

#-------------------------------------------------------------------

class ImageList(list):
    def __init__(self, imgList=[]):
        super(ImageList, self).__init__(imgList)
        self.UpdateLinks()

    def __delitem__(self, item):
        super(ImageList, self).__delitem__(item)
        self.UpdateAndRestrainLinks()

    def UpdateLinks(self):
        for imgPrev, imgNext in zip(self[:-1], self[1:]):
            imgPrev.next = imgNext
            imgNext.prev = imgPrev
            imgNext.numInSeries = imgPrev.numInSeries + 1

    def UpdateAndRestrainLinks(self):
        self[0].prev = None
        self[0].numInSeries = 1
        self[len(self)-1].next = None
        self.UpdateLinks()

#-------------------------------------------------------------------

def am_ph_2_re_im_cpu(img):
    img_ri = ImageExp(img.height, img.width, cmpRepr=Image.cmp['CRI'])
    if img.cmpRepr == Image.cmp['CRI']:
        img_ri.reIm = np.copy(img.reIm)
    else:
        img_ri.reIm = img.amPh.am * np.exp(1j * img.amPh.ph)
    return img_ri

# ---------------------------------------------------------------

def re_im_2_am_ph_cpu(img):
    img_ap = ImageExp(img.height, img.width, cmpRepr=Image.cmp['CAP'])
    if img.cmpRepr == Image.cmp['CAP']:
        img_ap.amPh.am = np.copy(img.amPh.am)
        img_ap.amPh.ph = np.copy(img.amPh.ph)
    else:
        img_ap.amPh.am = np.abs(img.reIm)
        img_ap.amPh.ph = np.angle(img.reIm)
    return img_ap

#-------------------------------------------------------------------

def PrepareImageMatrix(imgData, dimSize):
    imgArray = np.asarray(imgData)
    imgMatrix = np.reshape(imgArray, (-1, dimSize))
    if len(imgMatrix[imgMatrix < 0]) > 0:
        imgMatrix += np.abs(np.min(imgMatrix))
    # imgMatrix = np.abs(imgMatrix)         # some pixels can have negative values
    return imgMatrix

#-------------------------------------------------------------------

def FileToImage(fPath):
    import Dm3Reader as dm3
    img_data, px_dims = dm3.ReadDm3File(fPath)
    img_matrix = PrepareImageMatrix(img_data, img_data.shape[0])
    img = ImageExp(img_data.shape[0], img_data.shape[1], Image.cmp['CAP'])
    img.amPh.am = np.sqrt(img_matrix).astype(np.float32)
    img.UpdateBuffer()
    img.px_dim = px_dims[0]
    Image.px_dim_default = px_dims[0]
    return img

#-------------------------------------------------------------------

def ScaleImage(img, newMin, newMax):
    # currMin = np.delete(img, np.argwhere(img==0)).min()
    currMin = img.min()
    currMax = img.max()
    if currMin == currMax:
        return img
    imgScaled = (img - currMin) * (newMax - newMin) / (currMax - currMin) + newMin
    return imgScaled

#-------------------------------------------------------------------

def PrepareImageToDisplay(img, capVar, scale=True, log=False, color=False):
    dt = img.cmpRepr
    img.ReIm2AmPh()
    imgVar = img.amPh.am if capVar == Image.capVar['AM'] else img.amPh.ph
    img.ChangeComplexRepr(dt)

    if log:
        if np.min(imgVar) < 0:
            imgVar -= np.min(imgVar)
        if np.min(imgVar) == 0:
            imgVar += 1e-6
        imgVar = np.log10(imgVar)

    if scale:
        imgVar = ScaleImage(imgVar, 0.0, 255.0)

    if not color:
        imgToDisp = im.fromarray(imgVar.astype(np.uint8))
    else:
        imgVarCol = grayscale_to_rgb(imgVar)
        imgToDisp = im.fromarray(imgVarCol.astype(np.uint8), 'RGB')

    return imgToDisp

#-------------------------------------------------------------------

def gs_to_rgb(gs_val, rgb_cm):
    return rgb_cm[int(gs_val)]

#-------------------------------------------------------------------

def grayscale_to_rgb(gs_arr):
    gs_arr[gs_arr < 0] = 0
    gs_arr[gs_arr > 255] = 255

    step = 4
    inc_range = np.arange(0, 256, step)
    dec_range = np.arange(255, -1, -step)

    bcm1 = [[0, i, 255] for i in inc_range]
    gcm1 = [[0, 255, i] for i in dec_range]
    gcm2 = [[i, 255, 0] for i in inc_range]
    rcm1 = [[255, i, 0] for i in dec_range]
    # rcm2 = [[255, 0, i] for i in inc_range]
    # bcm2 = [[i, 0, 255] for i in dec_range]
    rgb_cm = bcm1 + gcm1 + gcm2 + rcm1
    rgb_cm = np.array(rgb_cm[:256])

    gs_arr_1d = gs_arr.reshape((1, -1))
    rgb_arr_1d = rgb_cm[list(gs_arr_1d.astype(np.uint8))]
    rgb_arr = rgb_arr_1d.reshape((gs_arr.shape[0], gs_arr.shape[1], 3))

    return rgb_arr

#-------------------------------------------------------------------

def DisplayAmpImage(img, scale=True, log=False):
    imgToDisp = PrepareImageToDisplay(img, Image.capVar['AM'], scale, log)
    imgToDisp.show()

# -------------------------------------------------------------------

def SaveAmpImage(img, fPath, scale=True, log=False, color=False):
    imgToSave = PrepareImageToDisplay(img, Image.capVar['AM'], scale, log, color)
    imgToSave.save(fPath)

#-------------------------------------------------------------------

def DisplayPhaseImage(img, scale=True, log=False):
    imgToDisp = PrepareImageToDisplay(img, Image.capVar['PH'], scale, log)
    imgToDisp.show()

# -------------------------------------------------------------------

def SavePhaseImage(img, fPath, scale=True, log=False, color=False):
    imgToSave = PrepareImageToDisplay(img, Image.capVar['PH'], scale, log, color)
    imgToSave.save(fPath)

# -------------------------------------------------------------------

def crop_am_ph_roi(img, coords):
    roi_h = coords[3] - coords[1]
    roi_w = coords[2] - coords[0]
    roi = ImageExp(roi_h, roi_w, img.cmpRepr, px_dim_sz=img.px_dim)

    roi.amPh.am[:] = img.amPh.am[coords[1]:coords[3], coords[0]:coords[2]]
    roi.amPh.ph[:] = img.amPh.ph[coords[1]:coords[3], coords[0]:coords[2]]
    roi.UpdateBuffer()

    if img.cos_phase is not None:
        roi.update_cos_phase()
    return roi

# -------------------------------------------------------------------

def crop_img_roi_tl(img, tl_pt, roi_dims):
    dt = img.cmpRepr
    img.AmPh2ReIm()
    roi = ImageExp(roi_dims[0], roi_dims[1], img.cmpRepr, px_dim_sz=img.px_dim)
    roi.reIm = img.reIm[tl_pt[0]:tl_pt[0] + roi_dims[0], tl_pt[1]:tl_pt[1] + roi_dims[1]]
    img.ChangeComplexRepr(dt)
    roi.ChangeComplexRepr(dt)
    roi.defocus = img.defocus
    return roi

# -------------------------------------------------------------------

def crop_img_roi_mid(img, mid_pt, roi_dims):
    dt = img.cmpRepr
    img.AmPh2ReIm()
    roi = ImageExp(roi_dims[0], roi_dims[1], img.cmpRepr, px_dim_sz=img.px_dim)
    half_dim1 = roi_dims[0] // 2
    half_dim2 = roi_dims[1] // 2
    roi.reIm = img.reIm[mid_pt[0] - half_dim1:mid_pt[0] + half_dim1, mid_pt[1] - half_dim2:mid_pt[1] + half_dim2]
    img.ChangeComplexRepr(dt)
    roi.ChangeComplexRepr(dt)
    roi.defocus = img.defocus
    return roi

# -------------------------------------------------------------------

def DetermineCropCoords(width, height, shift):
    dx, dy = shift
    if dx >= 0 and dy >= 0:
        coords = [dy, dx, height, width]
    elif dy < 0 <= dx:
        coords = [0, dx, height+dy, width]
    elif dx < 0 <= dy:
        coords = [dy, 0, height, width+dx]
    else:
        coords = [0, 0, height+dy, width+dx]
    return coords

#-------------------------------------------------------------------

def DetermineCropCoordsAfterRotation(imgDim, rotDim, angle):
    angleRad = Radians(angle % 90)
    newDim = int((1 / np.sqrt(2)) * imgDim / np.cos(angleRad - Radians(45)))
    origX = int(rotDim / 2 - newDim / 2)
    cropCoords = [origX] * 2 + [origX + newDim] * 2
    return cropCoords

#-------------------------------------------------------------------

def DetermineCropCoordsForNewDims(oldWidth, oldHeight, newWidth, newHeight):
    origX = (oldWidth - newWidth) // 2
    origY = (oldHeight - newHeight) // 2
    cropCoords = [ origX, origY, origX + newWidth, origY + newHeight ]
    return cropCoords

#-------------------------------------------------------------------

def GetCommonArea(coords1, coords2):
    # 1st way (lists)
    coords3 = []
    coords3[0:2] = [c1 if c1 > c2 else c2 for c1, c2 in zip(coords1[0:2], coords2[0:2])]
    coords3[2:4] = [c1 if c1 < c2 else c2 for c1, c2 in zip(coords1[2:4], coords2[2:4])]
    return coords3

    # # 2nd way (numpy arrays)
    # coords1Arr = np.array(coords1)
    # coords2Arr = np.array(coords2)
    # coords3Arr = np.zeros(4)
    # coords3Arr[0:2] = np.fromiter((np.where(c1 > c2, c1, c2) for c1, c2 in zip(coords1Arr[0:2], coords2Arr[0:2])))
    # coords3Arr[2:4] = np.fromiter((np.where(c1 > c2, c2, c1) for c1, c2 in zip(coords1Arr[2:4], coords2Arr[2:4])))
    # return list(coords3Arr)

#-------------------------------------------------------------------

def MakeSquareCoords(coords):
    height = coords[3] - coords[1]
    width = coords[2] - coords[0]
    halfDiff = abs(height - width) // 2
    dimFix = 1 if (height + width) % 2 else 0

    if height > width:
        # squareCoords = [coords[1] + halfDiff + dimFix, coords[0], coords[3] - halfDiff, coords[2]]
        squareCoords = [coords[0], coords[1] + halfDiff + dimFix, coords[2], coords[3] - halfDiff]
    else:
        # squareCoords = [coords[1], coords[0] + halfDiff + dimFix, coords[3], coords[2] - halfDiff]
        squareCoords = [coords[0] + halfDiff + dimFix, coords[1], coords[2] - halfDiff, coords[3]]
    return squareCoords

#-------------------------------------------------------------------

def ClearImageData(img):
    shape = img.reIm.shape
    img.reIm = np.zeros(shape, dtype=np.complex64)
    img.amPh.am = np.zeros(shape, dtype=np.float32)
    img.amPh.ph = np.zeros(shape, dtype=np.float32)

#-------------------------------------------------------------------

def copy_re_im_image(img):
    img_copy = ImageExp(img.height, img.width, cmpRepr=img.cmpRepr, defocus=img.defocus, num=img.numInSeries,
                        px_dim_sz=img.px_dim)
    img_copy.reIm = np.copy(img.amPh.reIm)
    return img_copy

#-------------------------------------------------------------------

def copy_am_ph_image(img):
    img_copy = ImageExp(img.height, img.width, cmpRepr=img.cmpRepr, defocus=img.defocus, num=img.numInSeries,
                        px_dim_sz=img.px_dim)
    img_copy.amPh.am = np.copy(img.amPh.am)
    img_copy.amPh.ph = np.copy(img.amPh.ph)
    return img_copy

#-------------------------------------------------------------------

def copy_image(img):
    if img.cmpRepr == Image.cmp['CRI']:
        img_copy = copy_re_im_image(img)
    else:
        img_copy = copy_am_ph_image(img)
    return img_copy

#-------------------------------------------------------------------

def create_imgexp_from_img(img):
    return copy_image(img)

#-------------------------------------------------------------------

def GetFirstImage(img):
    first = img
    while first.prev is not None:
        first = first.prev
    return first

#-------------------------------------------------------------------

def GetLastImage(img):
    last = img
    while last.next is not None:
        last = last.next
    return last

#-------------------------------------------------------------------

def CreateImageListFromFirstImage(img):
    imgList = ImageList()
    imgList.append(img)
    while img.next is not None:
        img = img.next
        imgList.append(img)
    return imgList

#-------------------------------------------------------------------

def CreateImageListFromImage(img, howMany):
    imgList = ImageList()
    imgList.append(img)
    for idx in range(howMany-1):
        img = img.next
        imgList.append(img)
    return imgList

#-------------------------------------------------------------------

def pad_array(arr, arr_pad, pads, pad_val):
    p_height, p_width = arr_pad.shape
    arr_pad[pads[0]:p_height - pads[1], pads[2]:p_width - pads[3]] = np.copy(arr)
    arr_pad[0:pads[0], :] = pad_val
    arr_pad[p_height - pads[1]:p_height, :] = pad_val
    arr_pad[:, 0:pads[2]] = pad_val
    arr_pad[:, p_width - pads[3]:p_width] = pad_val

#-------------------------------------------------------------------

def pad_img_with_buf(img, buf_sz, pad_value, dirs):
    if buf_sz == 0:
        return img

    pads = [ buf_sz if d in 'tblr' else 0 for d in dirs ]
    p_height = img.height + pads[0] + pads[1]
    p_width = img.width + pads[2] + pads[3]
    img.ReIm2AmPh()

    pad_img = ImageExp(p_height, p_width, img.cmpRepr, num=img.numInSeries, px_dim_sz=img.px_dim)
    pad_img.amPh.am = pad_img.amPh.am.astype(img.amPh.am.dtype)
    pad_img.amPh.ph = pad_img.amPh.ph.astype(img.amPh.ph.dtype)

    pad_array(img.amPh.am, pad_img.amPh.am, pads, pad_value)
    pad_array(img.amPh.ph, pad_img.amPh.ph, pads, pad_value)

    return pad_img

#-------------------------------------------------------------------

def pad_img_from_ref(img, ref_width, pad_value):
    buf_sz_tot = ref_width - img.width
    if buf_sz_tot <= 0:
        return img
    buf_sz = buf_sz_tot // 2

    if buf_sz_tot % 2 == 0:
        pads = 4 * [buf_sz]
    else:
        pads = [ (buf_sz + (1 if idx % 2 else 0)) for idx in range(4) ]

    p_height = img.height + pads[0] + pads[1]
    p_width = img.width + pads[2] + pads[3]
    img.ReIm2AmPh()

    pad_img = ImageExp(p_height, p_width, img.cmpRepr, num=img.numInSeries, px_dim_sz=img.px_dim)
    pad_img.amPh.am = pad_img.amPh.am.astype(img.amPh.am.dtype)
    pad_img.amPh.ph = pad_img.amPh.ph.astype(img.amPh.ph.dtype)

    pad_array(img.amPh.am, pad_img.amPh.am, pads, pad_value)
    pad_array(img.amPh.ph, pad_img.amPh.ph, pads, pad_value)

    return pad_img

#-------------------------------------------------------------------

def Radians(angle):
    return angle * np.pi / 180

# -------------------------------------------------------------------

def Degrees(angle):
    return angle * 180 / np.pi

#-------------------------------------------------------------------

def FillImageWithValue(img, value):
    img.ReIm2AmPh()
    img.amPh.am.fill(value)

#-------------------------------------------------------------------

def RemovePixelArtifacts(img, minThreshold=0.7, maxThreshold=1.3):
    dt = img.cmpRepr
    img.ReIm2AmPh()

    arr = np.copy(img.amPh.am)
    arrAvg = np.average(arr)

    badPixels1 = np.where(arr >= (maxThreshold * arrAvg))
    badPixels2 = np.where(arr <= (minThreshold * arrAvg))
    arrCorr1 = arr * (arr < (maxThreshold * arrAvg))
    arrCorr2 = arrCorr1 * (arrCorr1 > (minThreshold * arrAvg))

    for y, x in zip(badPixels1[0], badPixels1[1]):
        arrCorr2[y, x] = arrAvg
    for y, x in zip(badPixels2[0], badPixels2[1]):
        arrCorr2[y, x] = arrAvg

    img.amPh.am = np.copy(arrCorr2)
    img.buffer = np.copy(arrCorr2)
    img.ChangeComplexRepr(dt)

#-------------------------------------------------------------------

def flip_image_h(img):
    img.amPh.am = np.fliplr(img.amPh.am)
    img.amPh.ph = np.fliplr(img.amPh.ph)

#-------------------------------------------------------------------

def flip_image_v(img):
    img.amPh.am = np.flipud(img.amPh.am)
    img.amPh.ph = np.flipud(img.amPh.ph)

#-------------------------------------------------------------------

def FFT(img):
    dt = img.cmpRepr
    img.AmPh2ReIm()

    fft = ImageExp(img.height, img.width, Image.cmp['CRI'])
    fft.reIm = np.fft.fft2(img.reIm)

    img.ChangeComplexRepr(dt)
    fft.ChangeComplexRepr(dt)
    return fft

#-------------------------------------------------------------------

def IFFT(fft):
    dt = fft.cmpRepr
    fft.AmPh2ReIm()

    ifft = ImageExp(fft.height, fft.width, Image.cmp['CRI'])
    ifft.reIm = np.fft.ifft2(fft.reIm)
    # ifft.reIm = sp.fft.ifft2(fft.reIm)
    # ifft.reIm = sp.fftpack.ifft2(fft.reIm)        # legacy

    fft.ChangeComplexRepr(dt)
    ifft.ChangeComplexRepr(dt)
    return ifft

#-------------------------------------------------------------------

def FFT2Diff(fft):
    dt = fft.cmpRepr
    fft.AmPh2ReIm()

    diff = ImageExp(fft.height, fft.width, cmpRepr=Image.cmp['CRI'])
    mid = diff.width // 2
    diff.reIm[:mid, :mid] = fft.reIm[mid:, mid:]
    diff.reIm[:mid, mid:] = fft.reIm[mid:, :mid]
    diff.reIm[mid:, :mid] = fft.reIm[:mid, mid:]
    diff.reIm[mid:, mid:] = fft.reIm[:mid, :mid]

    fft.ChangeComplexRepr(dt)
    diff.ChangeComplexRepr(dt)
    return diff

#-------------------------------------------------------------------

def Diff2FFT(diff):
    return FFT2Diff(diff)

#-------------------------------------------------------------------

def frange(x, y, jump):
    while x < y:
        yield x
        x += jump

#-------------------------------------------------------------------

def CalcCrossCorrFun(img1, img2):
    fft1 = FFT(img1)
    fft2 = FFT(img2)

    fft1.ReIm2AmPh()
    fft2.ReIm2AmPh()

    fft3 = ImageExp(fft1.height, fft1.width, Image.cmp['CAP'])
    fft1.amPh = conj_am_ph_matrix(fft1.amPh)
    fft3.amPh = mult_am_ph_matrices(fft1.amPh, fft2.amPh)
    fft3.amPh.am = np.sqrt(fft3.amPh.am)			# mcf ON

    # ---- ccf ----
    # fft3.amPh.am = fft1.amPh.am * fft2.amPh.am
    # fft3.amPh.ph = -fft1.amPh.ph + fft2.amPh.ph
    # ---- mcf ----
    # fft3.amPh.am = np.sqrt(fft1.amPh.am * fft2.amPh.am)
    # fft3.amPh.ph = -fft1.amPh.ph + fft2.amPh.ph

    ccf = IFFT(fft3)
    ccf = FFT2Diff(ccf)
    ccf.ReIm2AmPh()
    return ccf

#-------------------------------------------------------------------

def GetShift(ccf):
    dt = ccf.cmpRepr
    ccf.ReIm2AmPh()

    ccfMidXY = np.array(ccf.amPh.am.shape) // 2
    ccfMaxXY = np.array(np.unravel_index(np.argmax(ccf.amPh.am), ccf.amPh.am.shape))
    shift = tuple(ccfMidXY - ccfMaxXY)[::-1]

    ccf.ChangeComplexRepr(dt)
    return shift

#-------------------------------------------------------------------

def shift_array(arr, dx, dy, fval=0.0):
    if dx == 0 and dy == 0:
        return np.copy(arr)
    # arr_shifted = np.zeros(arr.shape, dtype=arr.dtype)
    arr_shifted = np.full(arr.shape, fval, dtype=arr.dtype)

    if dy == 0:
        if dx > 0:
            arr_shifted[:, dx:] = arr[:, :-dx]
        else:
            arr_shifted[:, :dx] = arr[:, -dx:]
        return arr_shifted

    if dx == 0:
        if dy > 0:
            arr_shifted[dy:, :] = arr[:-dy, :]
        else:
            arr_shifted[:dy, :] = arr[-dy:, :]
        return arr_shifted

    if dx > 0 and dy > 0:
        arr_shifted[dy:, dx:] = arr[:-dy, :-dx]
    elif dx > 0 > dy:
        arr_shifted[:dy, dx:] = arr[-dy:, :-dx]
    elif dx < 0 < dy:
        arr_shifted[dy:, :dx] = arr[:-dy, -dx:]
    else:
        arr_shifted[:dy, :dx] = arr[-dy:, -dx:]

    return arr_shifted

#-------------------------------------------------------------------

def shift_am_ph_image(img, shift):
    dx, dy = shift
    img_shifted = ImageExp(img.height, img.width, img.cmpRepr, px_dim_sz=img.px_dim)

    img_shifted.amPh.am = shift_array(img.amPh.am, dx, dy)
    img_shifted.amPh.ph = shift_array(img.amPh.ph, dx, dy)

    if img.cos_phase is not None:
        img_shifted.update_cos_phase()

    return img_shifted

#-------------------------------------------------------------------

def ShiftImage(img, shift):
    dt = img.cmpRepr
    img.AmPh2ReIm()
    dx, dy = shift

    img_shifted = ImageExp(img.height, img.width, img.cmpRepr)
    img_shifted.reIm = shift_array(img.reIm, dx, dy)

    img.ChangeComplexRepr(dt)
    img_shifted.ChangeComplexRepr(dt)
    return img_shifted

#-------------------------------------------------------------------

def det_Imin_Imax_from_contrast(dI, def_max=256.0):
    dImin = dI // 2 + 1
    dImax = dI - dImin
    Imin = def_max // 2 - dImin
    Imax = def_max // 2 + dImax
    return Imin, Imax

#-------------------------------------------------------------------

def update_image_bright_cont_gamma(img_src, brg=0, cnt=255, gam=1.0):
    Imin, Imax = det_Imin_Imax_from_contrast(cnt)

    # option 1 (c->b->g)
    # correct contrast
    img_scaled = ScaleImage(img_src, Imin, Imax)
    # correct brightness
    img_scaled += brg
    img_scaled[img_scaled < 0.0] = 0.0
    # correct gamma
    img_scaled **= gam
    img_scaled[img_scaled > 255.0] = 255.0

    # # option 2 (c->g->b)
    # # correct contrast
    # img_scaled = imsup.ScaleImage(img_src, Imin, Imax)
    # img_scaled[img_scaled < 0.0] = 0.0
    # # correct gamma
    # img_scaled **= gam
    # # correct brightness
    # img_scaled += brg
    # img_scaled[img_scaled < 0.0] = 0.0
    # img_scaled[img_scaled > 255.0] = 255.0
    return img_scaled

#-------------------------------------------------------------------

def save_arr_as_tiff(data, fname, log, color, brg=0, cnt=255, gam=1.0):
    if log:
        data[np.where(data <= 0)] = 1e-5
        data = np.log(data)
    data = update_image_bright_cont_gamma(data, brg, cnt, gam)
    if color:
        data = grayscale_to_rgb(data)
        data_to_save = im.fromarray(data.astype(np.uint8), 'RGB')
    else:
        data_to_save = im.fromarray(data.astype(np.uint8))
    data_to_save.save('{0}.tif'.format(fname))