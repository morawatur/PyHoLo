import numpy as np
from PIL import Image as im
from numba import cuda
import math, cmath
import Constants as const
import CudaConfig as ccfg
import CrossCorr as cc

#-------------------------------------------------------------------

class ComplexAmPhMatrix:
    mem = {'CPU': 0, 'GPU': 1}

    def __init__(self, height, width, memType=mem['CPU']):
        if memType == self.mem['CPU']:
            # self.am = np.empty((height, width), dtype=np.float32)
            # self.ph = np.empty((height, width), dtype=np.float32)
            self.am = np.zeros((height, width), dtype=np.float32)
            self.ph = np.zeros((height, width), dtype=np.float32)
        else:
            # self.am = cuda.device_array((height, width), dtype=np.float32)
            # self.ph = cuda.device_array((height, width), dtype=np.float32)
            self.am = cuda.to_device(np.zeros((height, width), dtype=np.float32))
            self.ph = cuda.to_device(np.zeros((height, width), dtype=np.float32))

    def __del__(self):
        del self.am
        del self.ph

    # def FillMatrix(self, amMat, phMat):
    #     self.am = amMat
    #     self.ph = phMat

#-------------------------------------------------------------------

def ConjugateAmPhMatrix(ap):
    blockDim, gridDim = ccfg.DetermineCudaConfig(ap.am.shape[0])
    apConj = ComplexAmPhMatrix(ap.am.shape[0], ap.am.shape[1], ComplexAmPhMatrix.mem['GPU'])
    ConjugateAmPhMatrix_dev[gridDim, blockDim](ap.am, ap.ph, apConj.am, apConj.ph)
    return apConj

#-------------------------------------------------------------------

@cuda.jit('void(float32[:, :], float32[:, :], float32[:, :], float32[:, :])')
def ConjugateAmPhMatrix_dev(am, ph, amConj, phConj):
    x, y = cuda.grid(2)
    if x >= am.shape[0] or y >= am.shape[1]:
        return
    amConj[x, y] = am[x, y]
    phConj[x, y] = -ph[x, y]

# -------------------------------------------------------------------

def MultAmPhMatrices(ap1, ap2):
    blockDim, gridDim = ccfg.DetermineCudaConfig(ap1.am.shape[0])
    apRes = ComplexAmPhMatrix(ap1.am.shape[0], ap1.am.shape[1], ComplexAmPhMatrix.mem['GPU'])
    MultAmPhMatrices_dev[gridDim, blockDim](ap1.am, ap1.ph, ap2.am, ap2.ph, apRes.am, apRes.ph)
    return apRes

# -------------------------------------------------------------------

@cuda.jit('void(float32[:, :], float32[:, :], float32[:, :], float32[:, :], float32[:, :], float32[:, :])')
def MultAmPhMatrices_dev(am1, ph1, am2, ph2, amRes, phRes):
    x, y = cuda.grid(2)
    if x >= am1.shape[0] or y >= am1.shape[1]:
        return
    amRes[x, y] = am1[x, y] * am2[x, y]
    phRes[x, y] = ph1[x, y] + ph2[x, y]

#-------------------------------------------------------------------

def cos_dev_wrapper(arr):
    blockDim, gridDim = ccfg.DetermineCudaConfig(arr.shape[0])
    cos_arr = cuda.to_device(np.zeros(arr.shape, dtype=np.float32))
    cos_dev[gridDim, blockDim](arr, cos_arr)
    return cos_arr

#-------------------------------------------------------------------

@cuda.jit('void(float32[:, :], float32[:, :])')
def cos_dev(arr, cos_arr):
    x, y = cuda.grid(2)
    if x >= arr.shape[0] or y >= arr.shape[1]:
        return
    cos_arr[x, y] = math.cos(arr[x, y])

#-------------------------------------------------------------------

class Image:
    cmp = {'CRI': 0, 'CAP': 1}
    capVar = {'AM': 0, 'PH': 1}
    criVar = {'RE': 0, 'IM': 1}
    mem = {'CPU': 0, 'GPU': 1}
    px_dim_default = 1.0

    def __init__(self, height, width, cmpRepr=cmp['CAP'], memType=mem['CPU'], defocus=0.0, num=1, px_dim_sz=-1.0):
        self.width = width
        self.height = height
        self.size = width * height
        self.name = ''
        if memType == self.mem['CPU']:
            # self.reIm = np.empty((height, width), dtype=np.complex64)
            self.reIm = np.zeros((height, width), dtype=np.complex64)
        elif memType == self.mem['GPU']:
            # self.reIm = cuda.device_array((height, width), dtype=np.complex64)
            self.reIm = cuda.to_device(np.zeros((height, width), dtype=np.complex64))
        self.amPh = ComplexAmPhMatrix(height, width, memType)
        self.cmpRepr = cmpRepr
        self.memType = memType
        self.defocus = defocus
        self.numInSeries = num
        self.prev = None
        self.next = None
        self.px_dim = px_dim_sz
        self.bias = 0.0
        self.gain = 255.0
        self.gamma = 1.0
        if px_dim_sz < 0:
            self.px_dim = self.px_dim_default
        # ClearImageData(self)

    def __del__(self):
        self.reIm = None
        self.amPh.am = None
        self.amPh.ph = None
        # cuda.current_context().deallocations.clear()

    def ClearGPUMemory(self):
        self.reIm = None
        self.amPh.am = None
        self.amPh.ph = None
        # cuda.current_context().deallocations.clear()

    def ChangeMemoryType(self, newType):
        if newType == self.mem['CPU']:
            self.MoveToCPU()
        elif newType == self.mem['GPU']:
            self.MoveToGPU()

    def MoveToGPU(self):
        if self.memType == self.mem['GPU']:
            return
        self.reIm = cuda.to_device(self.reIm)
        self.amPh.am = cuda.to_device(self.amPh.am)
        self.amPh.ph = cuda.to_device(self.amPh.ph)
        self.memType = self.mem['GPU']

    def MoveToCPU(self):
        if self.memType == self.mem['CPU']:
            return
        reIm1 = self.reIm.copy_to_host()
        am1 = self.amPh.am.copy_to_host()
        ph1 = self.amPh.ph.copy_to_host()
        self.reIm = None
        self.amPh.am = None
        self.amPh.ph = None
        # cuda.current_context().deallocations.clear()  # release GPU memory which is not referenced (gpu_pointer = None)
        self.reIm = np.copy(reIm1)
        self.amPh.am = np.copy(am1)
        self.amPh.ph = np.copy(ph1)
        self.memType = self.mem['CPU']

    def ChangeComplexRepr(self, newRepr):
        if newRepr == self.cmp['CAP']:
            self.ReIm2AmPh()
        elif newRepr == self.cmp['CRI']:
            self.AmPh2ReIm()

    def ReIm2AmPh(self):
        if self.cmpRepr == self.cmp['CAP']:
            return
        mt = self.memType
        self.MoveToGPU()
        blockDim, gridDim = ccfg.DetermineCudaConfigNew((self.height, self.width))
        ReIm2AmPh_dev[gridDim, blockDim](self.reIm, self.amPh.am, self.amPh.ph)
        self.cmpRepr = self.cmp['CAP']
        if mt == self.mem['CPU']:
            self.MoveToCPU()

    def AmPh2ReIm(self):
        if self.cmpRepr == self.cmp['CRI']:
            return
        mt = self.memType
        self.MoveToGPU()
        blockDim, gridDim = ccfg.DetermineCudaConfigNew((self.height, self.width))
        AmPh2ReIm_dev[gridDim, blockDim](self.amPh.am, self.amPh.ph, self.reIm)
        self.cmpRepr = self.cmp['CRI']
        if mt == self.mem['CPU']:
            self.MoveToCPU()

    def get_num_in_series_from_prev(self):
        if self.prev is not None:
            self.numInSeries = self.prev.numInSeries + 1

# -------------------------------------------------------------------

class ImageExp(Image):
    def __init__(self, height, width, cmpRepr=Image.cmp['CAP'], memType=Image.mem['CPU'], defocus=0.0, num=1, px_dim_sz=-1.0):
        super(ImageExp, self).__init__(height, width, cmpRepr, memType, defocus, num, px_dim_sz)
        self.parent = super(ImageExp, self)
        self.shift = [0, 0]
        self.rot = 0
        self.cos_phase = None
        if self.memType == self.mem['CPU']:
            self.buffer = np.zeros(self.amPh.am.shape, dtype=np.float32)
        else:
            self.buffer = cuda.to_device(np.zeros(self.amPh.am.shape, dtype=np.float32))

    def __del__(self):
        super(ImageExp, self).__del__()
        # del self.buffer
        self.buffer = None
        self.cos_phase = None
        # cuda.current_context().deallocations.clear()

    def LoadAmpData(self, ampData):
        self.amPh.am = np.copy(ampData)
        self.buffer = np.copy(ampData)

    def LoadPhsData(self, phsData):
        self.amPh.ph = np.copy(phsData)

    def UpdateBuffer(self):
        if self.memType == self.mem['CPU']:
            self.buffer = np.copy(self.amPh.am)
        else:
            self.buffer.copy_to_device(self.amPh.am)

    def UpdateImageFromBuffer(self):
        if self.memType == self.mem['CPU']:
            self.amPh.am = np.copy(self.buffer)
        else:
            self.amPh.am = cuda.device_array(self.buffer.shape, dtype=np.float32)
            self.amPh.am.copy_to_device(self.buffer)

    def MoveToGPU(self):
        if self.memType == self.mem['GPU']:
            return
        super(ImageExp, self).MoveToGPU()
        self.buffer = cuda.to_device(self.buffer)
        if self.cos_phase is not None:
            self.cos_phase = cuda.to_device(self.cos_phase)

    def MoveToCPU(self):
        if self.memType == self.mem['CPU']:
            return
        super(ImageExp, self).MoveToCPU()
        buf = self.buffer.copy_to_host()
        self.buffer = None
        # cuda.current_context().deallocations.clear()
        self.buffer = np.copy(buf)
        if self.cos_phase is not None:
            cos_phs = self.cos_phase.copy_to_host()
            self.cos_phase = None
            self.cos_phase = np.copy(cos_phs)

    def ReIm2AmPh(self):
        if self.cmpRepr == self.cmp['CAP']:
            return
        super(ImageExp, self).ReIm2AmPh()
        self.UpdateBuffer()

    def update_cos_phase(self):
        if self.memType == self.mem['CPU']:
            self.cos_phase = np.cos(self.amPh.ph)
        else:
            self.cos_phase = cos_dev_wrapper(self.amPh.ph)

#-------------------------------------------------------------------

class ImageList(list):
    def __init__(self, imgList=[]):
        super(ImageList, self).__init__(imgList)
        self.UpdateLinks()

    def UpdateLinks(self):
        for imgPrev, imgNext in zip(self[:-1], self[1:]):
            imgPrev.next = imgNext
            imgNext.prev = imgPrev
            imgNext.numInSeries = imgPrev.numInSeries + 1

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

@cuda.jit('void(complex64[:, :], float32[:, :], float32[:, :])')
def ReIm2AmPh_dev(reIm, am, ph):
    x, y = cuda.grid(2)
    if x >= reIm.shape[0] or y >= reIm.shape[1]:
        return
    # am[x, y] = abs(reIm[x, y])
    # ph[x, y] = cm.phase(reIm[x, y])
    am[x, y], ph[x, y] = cmath.polar(reIm[x, y])

#-------------------------------------------------------------------

@cuda.jit('void(float32[:, :], float32[:, :], complex64[:, :])')
def AmPh2ReIm_dev(am, ph, reIm):
    x, y = cuda.grid(2)
    if x >= am.shape[0] or y >= am.shape[1]:
        return
    # reIm[x, y] = am[x, y] * cm.cos(ph[x, y]) + 1j * am[x, y] * cm.sin(ph[x, y])
    reIm[x, y] = cmath.rect(am[x, y], ph[x, y])

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
    import Dm3Reader3 as dm3
    imgData = dm3.ReadDm3File(fPath)
    imgMatrix = PrepareImageMatrix(imgData, const.dimSize)
    img = ImageExp(const.dimSize, const.dimSize, Image.cmp['CAP'], Image.mem['CPU'])
    img.amPh.am = np.sqrt(imgMatrix).astype(np.float32)
    img.UpdateBuffer()
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

# zrobic wersje na GPU
def ScaleAmpImages(images):
    amMax = 0.0
    amMin = cc.FindMaxInImage(images[0])
    # amMin = np.max(images[0].amPh.am)

    for img in images:
        amMaxCurr = cc.FindMaxInImage(img)
        amMinCurr = cc.FindMinInImage(img)
        # amMaxCurr = np.max(img.amPh.am)
        # amMinCurr = np.min(img.amPh.am)
        if amMaxCurr >= amMax:
            amMax = amMaxCurr
        if amMinCurr <= amMin:
            amMin = amMinCurr

    for img in images:
        img.MoveToCPU()     # !!!
        img.amPh.am = ScaleImage(img.amPh.am, amMin, amMax)
        img.MoveToGPU()

#-------------------------------------------------------------------

# should handle also GPU images
def PrepareImageToDisplay(img, capVar, log=False, color=False):
    dt = img.cmpRepr
    img.ReIm2AmPh()
    imgVar = img.amPh.am if capVar == Image.capVar['AM'] else img.amPh.ph
    img.ChangeComplexRepr(dt)

    if log:
        imgVar = np.log10(imgVar)

    imgVarScaled = ScaleImage(imgVar, 0.0, 255.0)

    if not color:
        imgToDisp = im.fromarray(imgVarScaled.astype(np.uint8))
    else:
        imgVarCol = grayscale_to_rgb(imgVarScaled)
        imgToDisp = im.fromarray(imgVarCol.astype(np.uint8), 'RGB')

    return imgToDisp

#-------------------------------------------------------------------

def gs_to_rgb(gs_val, rgb_cm):
    return rgb_cm[int(gs_val)]

#-------------------------------------------------------------------

def grayscale_to_rgb(gs_arr):
    gs_arr[gs_arr < 0] = 0
    gs_arr[gs_arr > 255] = 255

    step = 6
    inc_range = np.arange(0, 256, step)
    dec_range = np.arange(255, -1, -step)

    bcm1 = [[0, i, 255] for i in inc_range]
    gcm1 = [[0, 255, i] for i in dec_range]
    gcm2 = [[i, 255, 0] for i in inc_range]
    rcm1 = [[255, i, 0] for i in dec_range]
    rcm2 = [[255, 0, i] for i in inc_range]
    bcm2 = [[i, 0, 255] for i in dec_range]
    rgb_cm = bcm1 + gcm1 + gcm2 + rcm1 + rcm2 + bcm2
    rgb_cm = np.array(rgb_cm[:256])

    gs_arr_1d = gs_arr.reshape((1, -1))
    rgb_arr_1d = rgb_cm[list(gs_arr_1d.astype(np.uint8))]
    rgb_arr = rgb_arr_1d.reshape((gs_arr.shape[0], gs_arr.shape[1], 3))

    return rgb_arr

#-------------------------------------------------------------------

def DisplayAmpImage(img, log=False):
    mt = img.memType
    img.MoveToCPU()
    imgToDisp = PrepareImageToDisplay(img, Image.capVar['AM'], log)
    imgToDisp.show()
    img.ChangeMemoryType(mt)

# -------------------------------------------------------------------

def SaveAmpImage(img, fPath, log=False, color=False):
    mt = img.memType
    img.MoveToCPU()
    imgToSave = PrepareImageToDisplay(img, Image.capVar['AM'], log, color)
    imgToSave.save(fPath)
    img.ChangeMemoryType(mt)

#-------------------------------------------------------------------

def DisplayPhaseImage(img, log=False):
    mt = img.memType
    img.MoveToCPU()
    imgToDisp = PrepareImageToDisplay(img, Image.capVar['PH'], log)
    imgToDisp.show()
    img.ChangeMemoryType(mt)

# -------------------------------------------------------------------

def SavePhaseImage(img, fPath, log=False, color=False):
    mt = img.memType
    img.MoveToCPU()
    imgToSave = PrepareImageToDisplay(img, Image.capVar['PH'], log, color)
    imgToSave.save(fPath)
    img.ChangeMemoryType(mt)

# -------------------------------------------------------------------

def crop_am_ph_roi(img, coords):
    mt = img.memType
    img.MoveToGPU()

    roi_h = coords[3] - coords[1]
    roi_w = coords[2] - coords[0]
    roi = ImageExp(roi_h, roi_w, img.cmpRepr, img.memType)
    top_left_d = cuda.to_device(np.array(coords[:2], dtype=np.int32))

    block_dim, grid_dim = ccfg.DetermineCudaConfigNew((roi_h, roi_w))
    CropImageROICoords_dev[grid_dim, block_dim](img.amPh.am, roi.amPh.am, top_left_d)
    CropImageROICoords_dev[grid_dim, block_dim](img.amPh.ph, roi.amPh.ph, top_left_d)

    img.ChangeMemoryType(mt)
    roi.ChangeMemoryType(mt)
    return roi

# -------------------------------------------------------------------

def crop_am_ph_roi_cpu(img, coords):
    img.MoveToCPU()

    roi_h = coords[3] - coords[1]
    roi_w = coords[2] - coords[0]
    roi = ImageExp(roi_h, roi_w, img.cmpRepr, img.memType)

    roi.amPh.am[:] = img.amPh.am[coords[1]:coords[3], coords[0]:coords[2]]
    roi.amPh.ph[:] = img.amPh.ph[coords[1]:coords[3], coords[0]:coords[2]]
    roi.UpdateBuffer()

    if img.cos_phase is not None:
        roi.update_cos_phase()
    return roi

# -------------------------------------------------------------------

def CropImageROICoords(img, coords):
    roiHeight = coords[3] - coords[1]
    roiWidth = coords[2] - coords[0]
    dt = img.cmpRepr
    img.AmPh2ReIm()
    roi = Image(roiHeight, roiWidth, img.cmpRepr, Image.mem['GPU'])
    topLeft_d = cuda.to_device(np.array(coords[:2], dtype=np.int32))
    blockDim, gridDim = ccfg.DetermineCudaConfigNew((roiHeight, roiWidth))
    CropImageROICoords_dev[gridDim, blockDim](img.reIm, roi.reIm, topLeft_d)
    img.ChangeComplexRepr(dt)
    roi.ChangeComplexRepr(dt)
    roi.defocus = img.defocus           # !!!
    roi.numInSeries = img.numInSeries   # !!!
    return roi

# -------------------------------------------------------------------

# @cuda.jit('void(complex64[:, :], complex64[:, :], int32[:])')
@cuda.jit()
def CropImageROICoords_dev(img, roi, topLeft):
    rx, ry = cuda.grid(2)
    if rx >= roi.shape[0] or ry >= roi.shape[1]:
        return
    x0, y0 = topLeft
    # if coords[0] < x < coords[2] and coords[1] < y < coords[3]:

    roiIdx = ry * roi.shape[0] + rx
    imgIdx = roiIdx + ry * (img.shape[0] - roi.shape[0]) + img.shape[0] * y0 + x0

    y = imgIdx // img.shape[0]
    x = imgIdx % img.shape[0]

    roi[rx, ry] = img[x, y]


# -------------------------------------------------------------------

def CropImageROI(img, roiOrig, roiDims, isOrigTopLeft):
    dt = img.cmpRepr
    img.AmPh2ReIm()
    roi = Image(roiDims[0], roiDims[1], img.cmpRepr, Image.mem['GPU'])
    roiOrig_d = cuda.to_device(np.array(roiOrig))
    blockDim, gridDim = ccfg.DetermineCudaConfigNew(roiDims)
    if isOrigTopLeft:
        CropImageROITopLeft_dev[gridDim, blockDim](img.reIm, roi.reIm, roiOrig_d)
    else:
        CropImageROIMid_dev[gridDim, blockDim](img.reIm, roi.reIm, roiOrig_d)
    img.ChangeComplexRepr(dt)
    roi.ChangeComplexRepr(dt)
    roi.defocus = img.defocus   # !!!
    return roi

# -------------------------------------------------------------------

@cuda.jit('void(complex64[:, :], complex64[:, :], int32[:])')
def CropImageROITopLeft_dev(img, roi, rStart):
    rx, ry = cuda.grid(2)
    if rx >= roi.shape[0] or ry >= roi.shape[1]:
        return
    x0, y0 = rStart

    roiIdx = ry * roi.shape[0] + rx
    imgIdx = roiIdx + ry * (img.shape[0] - roi.shape[0]) + img.shape[0] * y0 + x0

    y = imgIdx // img.shape[0]
    x = imgIdx % img.shape[0]

    roi[rx, ry] = img[x, y]

# -------------------------------------------------------------------

@cuda.jit('void(complex64[:, :], complex64[:, :], int32[:])')
def CropImageROIMid_dev(img, roi, rMid):
    rx, ry = cuda.grid(2)
    if rx >= roi.shape[0] or ry >= roi.shape[1]:
        return

    x0 = rMid[0] - roi.shape[0] // 2
    y0 = rMid[1] - roi.shape[1] // 2

    if x0 + rx < 0:
        x0 += img.shape[0]
    elif x0 + rx >= img.shape[0]:
        x0 -= img.shape[0]
    if y0 + ry < 0:
        y0 += img.shape[1]
    elif y0 + ry >= img.shape[1]:
        y0 -= img.shape[1]

    roiIdx = ry * roi.shape[0] + rx
    imgIdx = roiIdx + ry * (img.shape[0] - roi.shape[0]) + img.shape[0] * y0 + x0

    y = imgIdx // img.shape[0]
    x = imgIdx % img.shape[0]

    roi[rx, ry] = img[x, y]

# -------------------------------------------------------------------

def PasteROIToImage(img, roi, roiOrig):
    dt = img.cmpRepr
    img.AmPh2ReIm()
    imgNew = Image(img.height, img.width, Image.cmp['CRI'], Image.mem['GPU'])
    imgNew.reIm = img.reIm
    roiOrig_d = cuda.to_device(np.array(roiOrig))
    blockDim, gridDim = ccfg.DetermineCudaConfigNew(roi.reIm.shape)
    PasteROIToImage_dev[gridDim, blockDim](imgNew.reIm, roi.reIm, roiOrig_d)
    img.ChangeComplexRepr(dt)
    return imgNew

# -------------------------------------------------------------------

def PasteROIToImageTopLeft(img, roi, roiTopLeft, spot=False):
    img.MoveToGPU()
    roi.MoveToGPU()
    dt = img.cmpRepr
    imgNew = CopyImage(img)
    imgNew.AmPh2ReIm()
    roi.AmPh2ReIm()

    roiOrig = np.array(roiTopLeft) + np.array(roi.reIm.shape) // 2      # !!!
    roiOrig_d = cuda.to_device(roiOrig)
    blockDim, gridDim = ccfg.DetermineCudaConfigNew(roi.reIm.shape)
    if not spot:
        PasteROIToImage_dev[gridDim, blockDim](imgNew.reIm, roi.reIm, roiOrig_d)
    else:
        PasteSpotToImage_dev[gridDim, blockDim](imgNew.reIm, roi.reIm, roiOrig_d)
    imgNew.ChangeComplexRepr(dt)
    return imgNew

# -------------------------------------------------------------------

# prawie to samo co w CropImageROIMid_dev(...)
@cuda.jit('void(complex64[:, :], complex64[:, :], int32[:])')
def PasteROIToImage_dev(img, roi, rMid):
    rx, ry = cuda.grid(2)
    if rx >= roi.shape[0] or ry >= roi.shape[1]:
        return

    x0 = rMid[0] - roi.shape[0] // 2
    y0 = rMid[1] - roi.shape[1] // 2

    if x0 + rx < 0:
        x0 += img.shape[0]
    elif x0 + rx >= img.shape[0]:
        x0 -= img.shape[0]
    if y0 + ry < 0:
        y0 += img.shape[1]
    elif y0 + ry >= img.shape[1]:
        y0 -= img.shape[1]

    roiIdx = ry * roi.shape[0] + rx
    imgIdx = roiIdx + ry * (img.shape[0] - roi.shape[0]) + img.shape[0] * y0 + x0

    y = imgIdx // img.shape[0]
    x = imgIdx % img.shape[0]

    img[x, y] = roi[rx, ry]

# -------------------------------------------------------------------

@cuda.jit('void(complex64[:, :], complex64[:, :], int32[:])')
def PasteSpotToImage_dev(img, roi, rMid):
    rx, ry = cuda.grid(2)
    if rx >= roi.shape[0] or ry >= roi.shape[1]:
        return
    rw, rh = roi.shape
    radius = math.sqrt(float((rw // 2 - rx) ** 2 + (rh // 2 - ry) ** 2))
    if radius > roi.shape[0] // 2:
        return

    x0 = rMid[0] - roi.shape[0] // 2
    y0 = rMid[1] - roi.shape[1] // 2

    roiIdx = ry * roi.shape[0] + rx
    imgIdx = roiIdx + ry * (img.shape[0] - roi.shape[0]) + img.shape[0] * y0 + x0

    y = imgIdx // img.shape[0]
    x = imgIdx % img.shape[0]

    img[x, y] = roi[rx, ry]

#-------------------------------------------------------------------

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

def DetermineCropCoordsForNewWidth(oldWidth, newWidth):
    origX = (oldWidth - newWidth) // 2
    cropCoords = [ origX ] * 2 + [ origX + newWidth ] * 2
    return cropCoords

#-------------------------------------------------------------------

def DetermineCropCoordsForNewDims(oldWidth, oldHeight, newWidth, newHeight):
    origX = (oldWidth - newWidth) // 2
    origY = (oldHeight - newHeight) // 2
    cropCoords = [ origX, origY, origX + newWidth, origY + newHeight ]
    return cropCoords

#-------------------------------------------------------------------

def DetermineEqualCropCoords(biggerWidth, smallerWidth):
    halfDiff = (biggerWidth - smallerWidth) / 2
    coords = [halfDiff] * 2 + [biggerWidth - halfDiff] * 2
    return coords

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
    # diff = abs(height - width)
    halfDiff = abs(height - width) // 2
    dimFix = 1 if (height + width) % 2 else 0

    if height > width:
        # squareCoords = [0, halfDiff, width, height - halfDiff]
        squareCoords = [coords[1] + halfDiff + dimFix, coords[0], coords[3] - halfDiff, coords[2]]
        # squareCoords = [0, 0, width, height - diff]
    else:
        #squareCoords = [halfDiff, 0, width - halfDiff, height]
        squareCoords = [coords[1], coords[0] + halfDiff + dimFix, coords[3], coords[2] - halfDiff]
        # squareCoords = [0, 0, width - diff, height]
    return squareCoords

#-------------------------------------------------------------------

def ClearImageData(img):
    shape = img.reIm.shape
    if img.memType == Image.mem['CPU']:
        img.reIm = np.zeros(shape, dtype=np.complex64)
        img.amPh.am = np.zeros(shape, dtype=np.float32)
        img.amPh.ph = np.zeros(shape, dtype=np.float32)
    elif img.memType == Image.mem['GPU']:
        img.reIm = cuda.to_device(np.zeros(shape, dtype=np.complex64))
        img.amPh.am = cuda.to_device(np.zeros(shape, dtype=np.float32))
        img.amPh.ph = cuda.to_device(np.zeros(shape, dtype=np.float32))

    # mt = img.memType
    # img.MoveToGPU()
    # img.reIm = cuda.device_array(shape, dtype=np.complex64)
    # img.amPh.am = cuda.device_array(shape, dtype=np.float32)
    # img.amPh.ph = cuda.device_array(shape, dtype=np.float32)
    # img.ChangeMemoryType(mt)

#-------------------------------------------------------------------

def copy_re_im_image(img):
    img.MoveToCPU()
    img_copy = ImageExp(img.height, img.width, cmpRepr=img.cmpRepr, memType=img.memType, defocus=img.defocus, num=img.numInSeries)
    img_copy.reIm = np.copy(img.amPh.reIm)

    if img.prev is not None:
        img_copy.prev = img.prev
    if img.next is not None:
        img_copy.next = img.next

    return img_copy

#-------------------------------------------------------------------

def copy_am_ph_image(img):
    img.MoveToCPU()
    img_copy = ImageExp(img.height, img.width, cmpRepr=img.cmpRepr, memType=img.memType,
                        defocus=img.defocus, num=img.numInSeries, px_dim_sz=img.px_dim)
    img_copy.amPh.am = np.copy(img.amPh.am)
    img_copy.amPh.ph = np.copy(img.amPh.ph)

    if img.prev is not None:
        img_copy.prev = img.prev
    if img.next is not None:
        img_copy.next = img.next

    return img_copy

#-------------------------------------------------------------------

def CopyImage(img):
    mt = img.memType
    dt = img.cmpRepr
    img.AmPh2ReIm()
    img.MoveToCPU()

    imgCopy = ImageExp(img.height, img.width, cmpRepr=img.cmpRepr, memType=img.memType, defocus=img.defocus, num=img.numInSeries)
    imgCopy.reIm = np.copy(img.reIm)

    if img.prev is not None:
        imgCopy.prev = img.prev
    if img.next is not None:
        imgCopy.next = img.next

    img.ChangeComplexRepr(dt)
    img.ChangeMemoryType(mt)
    imgCopy.ChangeComplexRepr(dt)
    imgCopy.ChangeMemoryType(mt)
    return imgCopy

#-------------------------------------------------------------------

def create_imgexp_from_img(img):
    img_exp = copy_am_ph_image(img)
    img_exp.UpdateBuffer()
    return img_exp

#-------------------------------------------------------------------

def CreateImageExpFromImage(img):
    img_exp = CopyImage(img)
    img_exp.ReIm2AmPh()
    img_exp.UpdateBuffer()
    return img_exp

#-------------------------------------------------------------------

def GetFirstImage(img):
    first = img
    while first.prev is not None:
        first = first.prev
    return first

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

def PadArray(arr, arr_pad, pads, pad_val):
    p_height, p_width = arr_pad.shape
    arr_pad[pads[0]:p_height - pads[1], pads[2]:p_width - pads[3]] = np.copy(arr)
    arr_pad[0:pads[0], :] = pad_val
    arr_pad[p_height - pads[1]:p_height, :] = pad_val
    arr_pad[:, 0:pads[2]] = pad_val
    arr_pad[:, p_width - pads[3]:p_width] = pad_val

#-------------------------------------------------------------------

def PadImage(img, bufSz, padValue, dirs):
    if bufSz == 0:
        return img

    pads = [ bufSz if d in 'tblr' else 0 for d in dirs ]
    pHeight = img.height + pads[0] + pads[1]
    pWidth = img.width + pads[2] + pads[3]
    # rbPadding = ltPadding if not img.height % 2 else ltPadding + 1
    mt = img.memType
    img.ReIm2AmPh()
    img.MoveToCPU()

    imgPadded = ImageExp(pHeight, pWidth, img.cmpRepr, img.memType, img.defocus, img.numInSeries, px_dim_sz=img.px_dim)
    PadArray(img.amPh.am, imgPadded.amPh.am, pads, padValue)
    PadArray(img.amPh.ph, imgPadded.amPh.ph, pads, padValue)

    resc_factor = pWidth / img.width
    imgPadded.px_dim *= resc_factor

    img.ChangeMemoryType(mt)
    return imgPadded

#-------------------------------------------------------------------

def pad_img_from_ref(img, ref_width, pad_value, dirs):
    buf_sz_tot = ref_width - img.width
    buf_sz = buf_sz_tot // 2

    if buf_sz_tot % 2 == 0:
        pads = [buf_sz if d in 'tblr' else 0 for d in dirs]
    else:
        pads = [(buf_sz + (1 if idx % 2 else 0)) if d in 'tblr' else 0 for d, idx in zip(dirs, range(len(dirs)))]

    p_height = img.height + pads[0] + pads[1]
    p_width = img.width + pads[2] + pads[3]

    padded_img = ImageExp(p_height, p_width, img.cmpRepr, img.memType, img.defocus, img.numInSeries)
    PadArray(img.amPh.am, padded_img.amPh.am, pads, pad_value)
    PadArray(img.amPh.ph, padded_img.amPh.ph, pads, pad_value)

    return padded_img

#-------------------------------------------------------------------

def PadImageBufferToNx512(img, padValue):
    dimFactor = 512
    imgBufHeight = img.buffer.shape[0]
    imgBufWidth = img.buffer.shape[1]
    pHeight = int(np.ceil(imgBufHeight / dimFactor) * dimFactor)
    pWidth = int(np.ceil(imgBufWidth / dimFactor) * dimFactor)
    ltPadding = (pHeight - imgBufHeight) // 2
    rbPadding = ltPadding if not imgBufHeight % 2 else ltPadding + 1
    mt = img.memType
    img.ReIm2AmPh()
    img.MoveToCPU()

    imgPadded = ImageExp(pHeight, pWidth, img.cmpRepr, img.memType, img.defocus, img.numInSeries)
    imgPadded.buffer[ltPadding:pHeight-rbPadding, ltPadding:pWidth-rbPadding] = img.buffer
    imgPadded.buffer[0:ltPadding, :] = padValue
    imgPadded.buffer[pHeight-rbPadding:pHeight, :] = padValue
    imgPadded.buffer[:, 0:ltPadding] = padValue
    imgPadded.buffer[:, pWidth-rbPadding:pWidth] = padValue

    img.ChangeMemoryType(mt)
    return imgPadded

#-------------------------------------------------------------------

def RotateImage(img, deltaPhi):
    mt = img.memType
    img.MoveToGPU()
    img.ReIm2AmPh()

    piBy4 = np.pi / 4
    deltaPhiRad = deltaPhi * np.pi / 180.0
    rMax = np.sqrt((img.width / 2.0) ** 2 + (img.height / 2.0) ** 2)
    if (deltaPhiRad // piBy4 + 1) % 2:
        angle = piBy4 - (deltaPhiRad % piBy4)
    else:
        angle = deltaPhiRad % piBy4
    rotWidth = int(np.ceil(2 * rMax * np.cos(angle)))
    rotHeight = rotWidth
    imgRotated = Image(rotHeight, rotWidth, img.cmpRepr, img.memType)
    filled = cuda.to_device(np.zeros(imgRotated.amPh.am.shape, dtype=np.int32))

    # rotation

    blockDim, gridDim = ccfg.DetermineCudaConfigNew(img.amPh.am.shape)
    RotateImage_dev[gridDim, blockDim](img.amPh.am, imgRotated.amPh.am, filled, deltaPhiRad)
    imgRotated.MoveToGPU()
    blockDim, gridDim = ccfg.DetermineCudaConfigNew(imgRotated.amPh.am.shape)
    InterpolateMissingPixels_dev[gridDim, blockDim](imgRotated.amPh.am, filled)

    imgRotated = CreateImageExpFromImage(imgRotated)
    img.ChangeMemoryType(mt)
    return imgRotated

#-------------------------------------------------------------------

# @cuda.jit('void(complex64[:], float32[:], float32[:])')
# def XY2RadPhi_dev(xy, r, phi):
#     cx, cy = cuda.grid(2)
#     r[cx, cy] = cmath.sqrt(xy[cx, cy][0] ** 2 + xy[cx, cy][1] ** 2)
#     phi[cx, cy] = cmath.arctan2(xy[cx, cy][1], xy[cx, cy][0])

#-------------------------------------------------------------------

@cuda.jit('void(float32[:, :], float32[:, :], int32[:, :], float32)')
def RotateImage_dev(img, imgRot, filled, deltaPhi):
    x0, y0 = cuda.grid(2)
    if x0 >= img.shape[0] or y0 >= img.shape[1]:
        return
    r0, phi0 = cmath.polar(complex(x0 - img.shape[0] / 2, y0 - img.shape[1] / 2))
    # z1 = cmath.rect(r0, phi0 - deltaPhi)
    # x1 = int(z1.real + imgRot.shape[0] / 2)
    # y1 = int(z1.imag + imgRot.shape[1] / 2)
    x1 = int(r0 * math.cos(phi0 - deltaPhi) + imgRot.shape[0] / 2)
    y1 = int(r0 * math.sin(phi0 - deltaPhi) + imgRot.shape[1] / 2)
    imgRot[y1, x1] = img[y0, x0]
    filled[y1, x1] = 1

#-------------------------------------------------------------------

@cuda.jit('void(float32[:, :], int32[:, :])')
def InterpolateMissingPixels_dev(img, filled):
    x, y = cuda.grid(2)
    if x >= img.shape[0] or y >= img.shape[1]:
        return

    if filled[y, x] > 0:
        return

    x1 = x - (x > 0)
    y1 = y - (y > 0)
    x2 = x + (x < img.shape[0]-1)
    y2 = y + (y < img.shape[1]-1)

    nHoodSum = 0.0
    nPixels = 0
    for yy in range(y1, y2):
        for xx in range(x1, x2):
            nHoodSum += img[yy, xx] * filled[yy, xx]
            nPixels += filled[yy, xx]

    # nHoodSum = img[y1, x] * filled[y1, x] +\
    #            img[y2, x] * filled[y2, x] +\
    #            img[y, x1] * filled[y, x1] +\
    #            img[y, x2] * filled[y, x2]
    # nPixels = filled[y1, x] + filled[y2, x] + filled[y, x1] + filled[y, x2]

    img[y, x] = nHoodSum / nPixels

#-------------------------------------------------------------------

def DetermineCropCoordsAfterRotation(imgDim, rotDim, angle):
    angleRad = Radians(angle % 90)
    newDim = int((1 / np.sqrt(2)) * imgDim / np.cos(angleRad - Radians(45)))
    origX = int(rotDim / 2 - newDim / 2)
    cropCoords = [origX] * 2 + [origX + newDim] * 2
    return cropCoords

#-------------------------------------------------------------------

def RotateImageAndCrop(img, angle):
    imgRot = RotateImage(img, angle)
    rotCropCoords = DetermineCropCoordsAfterRotation(img.width, imgRot.width, angle)
    imgRotCrop = CropImageROICoords(imgRot, rotCropCoords)
    return imgRotCrop

#-------------------------------------------------------------------

def Radians(angle):
    return angle * np.pi / 180

# -------------------------------------------------------------------

def Degrees(angle):
    return angle * 180 / np.pi

#-------------------------------------------------------------------

def MagnifyImage(img, factor):
    img.MoveToGPU()
    img.ReIm2AmPh()
    magHeight = int(factor * img.height)
    magWidth = int(factor * img.width)
    imgScaled = Image(magHeight, magWidth, img.cmpRepr, img.memType)
    filled = cuda.to_device(np.zeros(imgScaled.amPh.am.shape, dtype=np.int32))

    # magnification

    blockDim, gridDim = ccfg.DetermineCudaConfigNew(img.amPh.am.shape)
    MagnifyImage_dev[gridDim, blockDim](img.amPh.am, imgScaled.amPh.am, filled, factor)

    imgScaled.MoveToGPU()
    blockDim, gridDim = ccfg.DetermineCudaConfigNew(imgScaled.amPh.am.shape)
    InterpolateMissingPixels_dev[gridDim, blockDim](imgScaled.amPh.am, filled)

    return imgScaled

#-------------------------------------------------------------------

@cuda.jit('void(float32[:, :], float32[:, :], int32[:, :], float32)')
def MagnifyImage_dev(img, imgMag, filled, factor):
    x0, y0 = cuda.grid(2)
    if x0 >= img.shape[0] or y0 >= img.shape[1]:
        return
    x1 = int(factor * x0)
    y1 = int(factor * y0)
    imgMag[y1, x1] = img[y0, x0]
    filled[y1, x1] = 1

#-------------------------------------------------------------------

def FillImageWithValue(img, value):
    img.ReIm2AmPh()
    img.MoveToCPU()
    img.amPh.am.fill(value)

# -------------------------------------------------------------

def JoinImagesH(images):
    width0 = images[0].width
    height0 = images[0].height
    joinedImage = Image(height0, len(images) * width0)
    for img, idx in zip(images, range(len(images))):
        roiOrig = [0, idx * width0]
        joinedImage = PasteROIToImageTopLeft(joinedImage, img, roiOrig)
    return joinedImage

# -------------------------------------------------------------

def JoinImagesV(images):
    width0 = images[0].width
    height0 = images[0].height
    joinedImage = Image(len(images) * height0, width0)
    for img, idx in zip(images, range(len(images))):
        roiOrig = [idx * height0, 0]
        joinedImage = PasteROIToImageTopLeft(joinedImage, img, roiOrig)
    return joinedImage

# -------------------------------------------------------------

def JoinImages(images, imgsInOneDim):
    imagesH = ImageList()
    for idx in range(imgsInOneDim):
        imagesH.append(JoinImagesH(images[idx * imgsInOneDim : (idx + 1) * imgsInOneDim]))
    imageV = JoinImagesV(imagesH)
    return imageV

# -------------------------------------------------------------

def LinkTwoImagesSmoothlyH(img1, img2):
    bufSz = int(0.1 * img1.width)
    img1p = PadImage(img1, bufSz, 0, '---r')
    img2p = PadImage(img2, bufSz, 0, '--l-')
    linkedImage = JoinImagesH([img1p, img2p])

    img1.ReIm2AmPh()
    img2.ReIm2AmPh()
    linkedImage.ReIm2AmPh()
    img1.MoveToCPU()
    img2.MoveToCPU()
    linkedImage.MoveToCPU()
    buffer = np.zeros((linkedImage.height, 2 * bufSz))
    mid = linkedImage.width // 2

    for row in range(linkedImage.height):
        lVal = img1.amPh.am[row, img1.width - 1]
        rVal = img2.amPh.am[row, 0]
        if lVal != rVal:
            buffer[row] = np.arange(lVal, rVal, (rVal - lVal) / (2 * bufSz), dtype=np.float32)[:2 * bufSz]
        else:
            buffer[row].fill(lVal)

    linkedImage.amPh.am[:, mid - bufSz:mid + bufSz] = np.copy(buffer)
    img1.MoveToGPU()
    img2.MoveToGPU()
    linkedImage.MoveToGPU()

    return linkedImage

# -------------------------------------------------------------

def LinkTwoImagesSmoothlyV(img1, img2):
    bufSz = int(0.1 * img1.height)
    img1p = PadImage(img1, bufSz, 0, '-b--')
    img2p = PadImage(img2, bufSz, 0, 't---')
    linkedImage = JoinImagesV([img1p, img2p])

    img1.ReIm2AmPh()
    img2.ReIm2AmPh()
    linkedImage.ReIm2AmPh()
    img1.MoveToCPU()
    img2.MoveToCPU()
    linkedImage.MoveToCPU()
    buffer = np.zeros((2 * bufSz, linkedImage.width))
    print(buffer.shape)
    mid = linkedImage.height // 2

    for col in range(linkedImage.width):
        tVal = img1.amPh.am[img1.height-1, col]
        bVal = img2.amPh.am[0, col]
        if tVal != bVal:
            buffer[:, col] = np.arange(tVal, bVal, (bVal - tVal) / (2 * bufSz), dtype=np.float32)[:2 * bufSz]
        else:
            buffer[:, col].fill(tVal)

    linkedImage.amPh.am[mid - bufSz:mid + bufSz, :] = np.copy(buffer)
    img1.MoveToGPU()
    img2.MoveToGPU()
    linkedImage.MoveToGPU()

    return linkedImage

#-------------------------------------------------------------------

# Funkcja zmiany limitow kontrastu polega wlasciwie na tym samym (trzeba tylko dac dwa warunki: gorny i dolny)
def RemovePixelArtifacts(img, minThreshold=0.7, maxThreshold=1.3):
    mt = img.memType
    dt = img.cmpRepr
    img.ReIm2AmPh()
    img.MoveToCPU()

    arr = np.copy(img.amPh.am)
    arrAvg = np.average(arr)

    badPixels1 = np.where(arr >= (maxThreshold * arrAvg))
    badPixels2 = np.where(arr <= (minThreshold * arrAvg))
    arrCorr1 = arr * (arr < (maxThreshold * arrAvg))
    arrCorr2 = arrCorr1 * (arrCorr1 > (minThreshold * arrAvg))
    # print('Registered {0} bad pixels'.format(len(badPixelIndices[0])))

    for y, x in zip(badPixels1[0], badPixels1[1]):
        # arrCorr2[y, x] = np.sum(arrCorr2[y-1:y+2, x-1:x+2]) / 8.0
        arrCorr2[y, x] = arrAvg
    for y, x in zip(badPixels2[0], badPixels2[1]):
        # arrCorr2[y, x] = np.sum(arrCorr2[y-1:y+2, x-1:x+2]) / 8.0
        arrCorr2[y, x] = arrAvg

    img.amPh.am = np.copy(arrCorr2)
    img.buffer = np.copy(arrCorr2)
    img.ChangeMemoryType(mt)
    img.ChangeComplexRepr(dt)

#-------------------------------------------------------------------

def move_to_ri_cpu(img):
    mt = img.memType
    dt = img.cmpRepr
    img.AmPh2ReIm()
    img.MoveToCPU()
    return mt, dt

#-------------------------------------------------------------------

def move_to_ap_cpu(img):
    mt = img.memType
    dt = img.cmpRepr
    img.ReIm2AmPh()
    img.MoveToCPU()
    return mt, dt

#-------------------------------------------------------------------

def restore_img_mt_dt(img, mt, dt):
    img.ChangeMemoryType(mt)
    img.ChangeComplexRepr(dt)

#-------------------------------------------------------------------

def flip_image_h(img):
    # mt, dt = move_to_ri_cpu(img)
    # re_im_flip = np.copy(np.fliplr(img.reIm))
    # img.reIm = np.copy(re_im_flip)
    am_flip = np.copy(np.fliplr(img.amPh.am))
    ph_flip = np.copy(np.fliplr(img.amPh.ph))
    img.amPh.am = np.copy(am_flip)
    img.amPh.ph = np.copy(ph_flip)
    # restore_img_mt_dt(img, mt, dt)

#-------------------------------------------------------------------

def flip_image_v(img):
    mt, dt = move_to_ri_cpu(img)
    re_im_flip = np.copy(np.flipud(img.reIm))
    img.reIm = np.copy(re_im_flip)
    restore_img_mt_dt(img, mt, dt)