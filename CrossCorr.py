import numpy as np
from accelerate.cuda import fft as cufft
from numba import cuda

import Constants as const
import CudaConfig as ccfg
import ImageSupport as imsup
import ArraySupport as arrsup
import Propagation as prop

#-------------------------------------------------------------------

def FFT(img):
    mt = img.memType
    dt = img.cmpRepr
    img.MoveToGPU()
    img.AmPh2ReIm()
    fft = imsup.ImageExp(img.height, img.width, imsup.Image.cmp['CRI'], imsup.Image.mem['GPU'])
    cufft.fft(img.reIm, fft.reIm)
    img.ChangeComplexRepr(dt)
    img.ChangeMemoryType(mt)
    return fft

#-------------------------------------------------------------------

def IFFT(fft):
    mt = fft.memType
    dt = fft.cmpRepr
    fft.MoveToGPU()
    fft.AmPh2ReIm()
    ifft = imsup.ImageExp(fft.height, fft.width, imsup.Image.cmp['CRI'], imsup.Image.mem['GPU'])
    cufft.ifft(fft.reIm, ifft.reIm)
    fft.ChangeComplexRepr(dt)
    fft.ChangeMemoryType(mt)
    return ifft

# -------------------------------------------------------------------

def FFT2Diff(fft):
    mt = fft.memType
    dt = fft.cmpRepr
    fft.MoveToGPU()
    fft.AmPh2ReIm()
    diff = imsup.ImageExp(fft.height, fft.width, imsup.Image.cmp['CRI'], imsup.Image.mem['GPU'])
    blockDim, gridDim = ccfg.DetermineCudaConfig(fft.width)
    FFT2Diff_dev[gridDim, blockDim](fft.reIm, diff.reIm, fft.width)
    diff.defocus = fft.defocus
    fft.ChangeComplexRepr(dt)
    fft.ChangeMemoryType(mt)
    return diff

# -------------------------------------------------------------------

def Diff2FFT(diff):
    return FFT2Diff(diff)

# -------------------------------------------------------------------

@cuda.jit('void(complex64[:, :], complex64[:, :], int32)')
def FFT2Diff_dev(fft, diff, dim):
    x, y = cuda.grid(2)
    if x >= fft.shape[0] or y >= fft.shape[1]:
        return
    diff[x, y] = fft[(x + dim // 2) % dim, (y + dim // 2) % dim]

#-------------------------------------------------------------------

def CalcCrossCorrFun(img1, img2):
    fft1 = FFT(img1)
    fft2 = FFT(img2)

    fft1.ReIm2AmPh()
    fft2.ReIm2AmPh()

    fft3 = imsup.Image(fft1.height, fft1.width, imsup.Image.cmp['CAP'], imsup.Image.mem['GPU'])
    fft1.amPh = imsup.ConjugateAmPhMatrix(fft1.amPh)
    fft3.amPh = imsup.MultAmPhMatrices(fft1.amPh, fft2.amPh)
    fft3.amPh.am = arrsup.CalcSqrtOfArray(fft3.amPh.am)			# mcf ON

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

def CalcAverageCrossCorrFun(img1, img2, nDiv):
    roiNR, roiNC  = img1.height // nDiv, img1.width // nDiv
    fragsToCorrelate1 = []
    fragsToCorrelate2 = []

    # mozna to wszystko urownoleglic (tak jak w CreateFragment())
    for y in range(0, nDiv):
        for x in range(0, nDiv):
            frag1 = imsup.CropImageROI(img1, (y * roiNR, x * roiNC), (roiNR, roiNC), 1)
            fragsToCorrelate1.append(frag1)

            frag2 = imsup.CropImageROI(img2, (y * roiNR, x * roiNC), (roiNR, roiNC), 1)
            fragsToCorrelate2.append(frag2)

    ccfAvg = imsup.Image(roiNR, roiNC, imsup.Image.cmp['CAP'], imsup.Image.mem['GPU'])

    for frag1, frag2 in zip(fragsToCorrelate1, fragsToCorrelate2):
        ccf = CalcCrossCorrFun(frag1, frag2)
        ccfAvg.amPh.am = arrsup.AddTwoArrays(ccfAvg.amPh.am, ccf.amPh.am)

    return ccfAvg

#-------------------------------------------------------------------

def frange(x, y, jump):
    while x < y:
        yield x
        x += jump

# -------------------------------------------------------------------

def FindMaxInImage(img):
    dimSize = img.width
    arrReduced = img.amPh.am
    blockDim, gridDim = ccfg.DetermineCudaConfigNew((dimSize // 2, dimSize // 2))
    while dimSize > 2:
        dimSize //= 2
        # arrReducedNew = cuda.device_array((dimSize, dimSize), dtype=np.float32)
        arrReducedNew = cuda.to_device(np.zeros((dimSize, dimSize), dtype=np.float32))
        ReduceArrayToFindMax_dev[gridDim, blockDim](arrReduced, arrReducedNew)
        arrReduced = arrReducedNew
        if gridDim[0] > 1:
            gridDim = [gridDim[0] // 2] * 2
        else:
            blockDim = [blockDim[0] // 2] * 2
    imgMax = np.max(arrReduced.copy_to_host())
    # imgMax = arrReduced.copy_to_host()[0, 0]
    return imgMax

# -------------------------------------------------------------------

# @cuda.jit()
@cuda.jit('void(float32[:, :], float32[:, :])')
def ReduceArrayToFindMax_dev(arr, arrRed):
    x, y = cuda.grid(2)
    if x >= arrRed.shape[0] or y >= arrRed.shape[1]:
        return
    arrRed[x, y] = max(arr[2*x, 2*y], max(arr[2*x, 2*y+1], max(arr[2*x+1, 2*y], arr[2*x+1, 2*y+1])))

# -------------------------------------------------------------------

def FindMinInImage(img):
    dimSize = img.width
    arrReduced = img.amPh.am
    blockDim, gridDim = ccfg.DetermineCudaConfig(dimSize // 2)
    while dimSize > 2:
        dimSize //= 2
        # arrReducedNew = cuda.device_array((dimSize, dimSize), dtype=np.float32)
        arrReducedNew = cuda.to_device(np.zeros((dimSize, dimSize), dtype=np.float32))
        ReduceArrayToFindMin_dev[gridDim, blockDim](arrReduced, arrReducedNew)
        arrReduced = arrReducedNew
        if gridDim[0] > 1:
            gridDim = [gridDim[0] // 2] * 2
        else:
            blockDim = [blockDim[0] // 2] * 2
    imgMin = np.min(arrReduced.copy_to_host())
    # imgMin = arrReduced.copy_to_host()[0, 0]
    return imgMin

# -------------------------------------------------------------------

# @cuda.jit()
@cuda.jit('void(float32[:, :], float32[:, :])')
def ReduceArrayToFindMin_dev(arr, arrRed):
    x, y = cuda.grid(2)
    if x >= arrRed.shape[0] or y >= arrRed.shape[1]:
        return
    arrRed[x, y] = min(arr[2*x, 2*y], min(arr[2*x, 2*y+1], min(arr[2*x+1, 2*y], arr[2*x+1, 2*y+1])))

# -------------------------------------------------------------------

def MaximizeMCF(img1, img2, dfStep0):
    # predefined defocus is given in m
    dfStep0 *= 1e6
    dfStepHalfRange = 0.8 * abs(dfStep0) 	# 4.8 um
    dfStepMin = dfStep0 - dfStepHalfRange 	# 1.2 um
    dfStepMax = dfStep0 + dfStepHalfRange  	# 10.8 um
    dfStepChange = 0.01 * abs(dfStep0)  	# 0.06 um
    return MaximizeMCFCore(img1, img2, 1, [(0, 0)], dfStepMin, dfStepMax, dfStepChange)

# -------------------------------------------------------------------

def MaximizeMCFCore(img1, img2, nDiv, fragCoords, dfStepMin, dfStepMax, dfStepChange):
    # defocus parameters are given in um
    dfStepMin, dfStepMax, dfStepChange = np.array([dfStepMin, dfStepMax, dfStepChange]) * 1e-6
    mcfMax = 0.0
    dfStepBest = img2.defocus - img1.defocus
    mcfBest = imsup.Image(img1.height, img1.width, imsup.Image.cmp['CRI'], imsup.Image.mem['GPU'])
    mcfBest.defocus = dfStepBest

    # mcfMaxPath = const.ccfMaxDir + const.ccfMaxName + str(img2.numInSeries) + '.txt'
    # mcfMaxFile = open(mcfMaxPath, 'w')

    for dfStep in frange(dfStepMin, dfStepMax, dfStepChange):
        ctf = prop.CalcTransferFunction(img1.width, img1.pxWidth, dfStep)
        # ctf.AmPh2ReIm()
        # ctf = Diff2FFT(ctf)
        img1Prop = prop.PropagateWave(img1, ctf)
        # mcf = CalcCrossCorrFun(img1Prop, img2)
        # mcf = CalcPartialCrossCorrFun(img1, img2, nDiv, fragCoords)
        mcf = CalcPartialCrossCorrFun(img1Prop, img2, nDiv, fragCoords)
        mcfMaxCurr = FindMaxInImage(mcf)
        # mcf.MoveToCPU()
        # mcfMaxCurr = np.max(mcf.amPh.am)
        # mcf.MoveToGPU()
        # mcfMaxFile.write('{0:.3f}\t{1:.3f}\n'.format(dfStep * 1e6, mcfMaxCurr))
        if mcfMaxCurr >= mcfMax:
            mcfMax = mcfMaxCurr
            dfStepBest = dfStep
            mcfBest = mcf

    # mcfMaxFile.close()
    mcfBest.defocus = dfStepBest
    print('Best defocus step = {0:.3f} um'.format(dfStepBest * 1e6))
    return mcfBest

#-------------------------------------------------------------------

def MaximizeMCFvsRotation(img1, img2, angleMin, angleMax):
    mcfMax = 0.0
    mcfBest = imsup.Image(img1.height, img1.width, imsup.Image.cmp['CRI'], imsup.Image.mem['GPU'])
    img2RotCropBest = imsup.Image(img1.height, img1.width, imsup.Image.cmp['CRI'], imsup.Image.mem['GPU'])
    bestAngle = 0
    for angle in range(angleMin, angleMax):
        img2Rot = imsup.RotateImage(img2, angle)
        rotCropCoords = imsup.DetermineCropCoordsAfterRotation(img2.width, img2Rot.width, angle)
        img2RotCrop = imsup.CropImageROICoords(img2Rot, rotCropCoords)
        # imsup.SaveAmpImage(img2RotCrop, 'img2Rot_{0}.png'.format(angle))
        cropCoords = imsup.DetermineCropCoordsForNewWidth(img1.width, img2RotCrop.width)
        img1Crop = imsup.CropImageROICoords(img1, cropCoords)
        # imsup.SaveAmpImage(img1Crop, 'img1Crop_{0}.png'.format(angle))
        mcf = CalcCrossCorrFun(img1Crop, img2RotCrop)
        # mcf.ReIm2AmPh()
        # mcf.MoveToCPU()
        # mcf.amPh.am[mcf.height // 2, :] = 0
        # mcf.amPh.am[:, mcf.width // 2] = 0
        # mcf.MoveToGPU()
        # imsup.SaveAmpImage(mcf, 'results/mcf/mcf{0:.0f}.png'.format(angle))
        mcfMaxCurr = FindMaxInImage(mcf)
        print(angle, mcfMaxCurr)
        if mcfMaxCurr >= mcfMax:
            mcfMax = mcfMaxCurr
            mcfBest = mcf
            img2RotCropBest = img2RotCrop
            bestAngle = angle

    print('Best rotation angle = {0:.2f} deg'.format(bestAngle))
    return mcfBest, img2RotCropBest

#-------------------------------------------------------------------

def GetShift(ccf):
    dt = ccf.cmpRepr
    mt = ccf.memType
    ccf.ReIm2AmPh()
    ccf.MoveToCPU()

    ccfMidXY = np.array(ccf.amPh.am.shape) // 2
    ccfMaxXY = np.array(np.unravel_index(np.argmax(ccf.amPh.am), ccf.amPh.am.shape))
    shift = tuple(ccfMidXY - ccfMaxXY)

    ccf.ChangeMemoryType(mt)
    ccf.ChangeComplexRepr(dt)
    return shift

#-------------------------------------------------------------------

def shift_am_ph_image(img, shift):
    mt = img.memType
    img.MoveToGPU()

    img_shifted = imsup.ImageExp(img.height, img.width, img.cmpRepr, img.memType)
    shift_d = cuda.to_device(np.array(shift))

    blockDim, gridDim = ccfg.DetermineCudaConfigNew(img.amPh.am.shape)
    ShiftImage_dev[gridDim, blockDim](img.amPh.am, img_shifted.amPh.am, shift_d, 0.0)
    ShiftImage_dev[gridDim, blockDim](img.amPh.ph, img_shifted.amPh.ph, shift_d, 0.0)

    if img.cos_phase is not None:
        img_shifted.update_cos_phase()

    img.ChangeMemoryType(mt)
    img_shifted.ChangeMemoryType(mt)
    return img_shifted

#-------------------------------------------------------------------

def ShiftImage(img, shift):
    mt = img.memType
    dt = img.cmpRepr
    img.MoveToGPU()
    img.AmPh2ReIm()

    imgShifted = imsup.Image(img.height, img.width, img.cmpRepr, img.memType)
    shift_d = cuda.to_device(np.array(shift))
    blockDim, gridDim = ccfg.DetermineCudaConfigNew(img.reIm.shape)
    ShiftImage_dev[gridDim, blockDim](img.reIm, imgShifted.reIm, shift_d, 0.0)

    img.ChangeComplexRepr(dt)
    img.ChangeMemoryType(mt)
    imgShifted.ChangeComplexRepr(dt)
    imgShifted.ChangeMemoryType(mt)
    return imgShifted

#-------------------------------------------------------------------

def ShiftImageAmpBuffer(img, shift):
    img.shift = [x + dx for x, dx in zip(img.shift, shift)]
    fillValue = np.max(img.amPh.am)
    img.MoveToGPU()
    imgShifted = imsup.Image(img.height, img.width, imsup.Image.cmp['CAP'], imsup.Image.mem['GPU'])
    shift_d = cuda.to_device(np.array(img.shift))
    blockDim, gridDim = ccfg.DetermineCudaConfigNew(img.buffer.shape)
    ShiftImage_dev[gridDim, blockDim](img.amPh.am, imgShifted.amPh.am, shift_d, fillValue)
    img.buffer.copy_to_device(imgShifted.amPh.am)
    img.MoveToCPU()

#-------------------------------------------------------------------

# def ShiftArray(arr, shift):
#     shift_d = cuda.to_device(np.array(shift))
#     arr_d = cuda.to_device(np.array(arr))
#     arrShifted_d = cuda.to_device(np.zeros(arr.shape, dtype=np.float32))
#     blockDim, gridDim = ccfg.DetermineCudaConfigNew(arr.shape)
#     ShiftImage_dev[gridDim, blockDim](arr_d, arrShifted_d, shift_d)
#     arr = arrShifted_d.to_host()

# -------------------------------------------------------------------

# @cuda.jit('void(complex64[:, :], complex64[:, :], int32[:])')
@cuda.jit()
def ShiftImage_dev(img, imgShifted, shift, fillValue=0.0):
    x, y = cuda.grid(2)
    if x >= img.shape[0] or y >= img.shape[1]:
        return
    dx, dy = shift

    if 0 <= x - dx < img.shape[0] and 0 <= y - dy < img.shape[1]:
        imgShifted[x, y] = img[x-dx, y-dy]
    else:
        # imgShifted[x, y] = 0.0
        imgShifted[x, y] = fillValue

# -------------------------------------------------------------------

def CalcPartialCrossCorrFun(img1, img2, nDiv, fragCoords):
    roiNR, roiNC = img1.height // nDiv, img1.width // nDiv
    fragsToCorrelate1 = []
    fragsToCorrelate2 = []

    for x, y in fragCoords:
        frag1 = imsup.CropImageROI(img1, (y * roiNR, x * roiNC), (roiNR, roiNC), 1)
        fragsToCorrelate1.append(frag1)

        frag2 = imsup.CropImageROI(img2, (y * roiNR, x * roiNC), (roiNR, roiNC), 1)
        fragsToCorrelate2.append(frag2)

    ccfAvg = imsup.Image(roiNR, roiNC, imsup.Image.cmp['CAP'], imsup.Image.mem['GPU'])

    for frag1, frag2 in zip(fragsToCorrelate1, fragsToCorrelate2):
        ccf = CalcCrossCorrFun(frag1, frag2)
        ccfAvg.amPh.am = arrsup.AddTwoArrays(ccfAvg.amPh.am, ccf.amPh.am)

    return ccfAvg

# -------------------------------------------------------------------

def CalcPartialCrossCorrFunUW(img1, img2, nDiv, fragCoords):
    roiNR, roiNC = img1.height // nDiv, img1.width // nDiv
    fragsToCorrelate1 = []
    fragsToCorrelate2 = []

    for x, y in fragCoords:
        frag1 = imsup.CropImageROI(img1, (y * roiNR, x * roiNC), (roiNR, roiNC), 1)
        fragsToCorrelate1.append(frag1)

        frag2 = imsup.CropImageROI(img2, (y * roiNR, x * roiNC), (roiNR, roiNC), 1)
        fragsToCorrelate2.append(frag2)

    shifts = np.zeros((nDiv, nDiv), dtype=np.complex)
    # fragsToJoin = imsup.ImageList()
    for (x, y), frag1, frag2 in zip(fragCoords, fragsToCorrelate1, fragsToCorrelate2):
        ccf = CalcCrossCorrFun(frag1, frag2)
        shift = GetShift(ccf)
        shifts[y, x] = np.complex(shift[1], shift[0])
        # frag2Shifted = ShiftImage(frag2, shift)
        # fragsToJoin.append(frag2Shifted)

    # shifts2dArray = np.array(shifts)
    shifts2dArray = np.copy(shifts)
    # img2Unwarped = imsup.JoinImages(fragsToJoin, nDiv)
    return shifts2dArray

# -------------------------------------------------------------------

def align_images_simple(img1, img2):
    common_coords = [0, 0, img1.height, img1.width]
    ccf = CalcCrossCorrFun(img1, img2)
    shift = GetShift(ccf)
    shifted_img2 = shift_am_ph_image(img2, shift)
    crop_coords = imsup.DetermineCropCoords(img2.height, img2.width, shift)
    common_coords = imsup.GetCommonArea(common_coords, crop_coords)
    square_coords = imsup.MakeSquareCoords(common_coords)
    aligned_img1 = imsup.CropImageROICoords(img1, square_coords)
    aligned_img2 = imsup.CropImageROICoords(shifted_img2, square_coords)
    return aligned_img1, aligned_img2

# -------------------------------------------------------------------

# align whole images with defocus adjustment
def AlignTwoImages(img1, img2, dfPars):
    commonCoords = [0, 0, img1.height, img1.width]
    mcfBest = MaximizeMCFCore(img1, img2, 1, [(0, 0)], dfPars[0], dfPars[1], dfPars[2])
    # ---
    # mcfBest.ReIm2AmPh()
    # mcfBest.MoveToCPU()
    # mcfBest.amPh.am[mcfBest.height // 2, :] = 0
    # mcfBest.amPh.am[:, mcfBest.width // 2] = 0
    # mcfBest.MoveToGPU()
    # ---
    imsup.SaveAmpImage(mcfBest, 'mcf.png')
    shift = GetShift(mcfBest)
    img2Shifted = ShiftImage(img2, shift)
    cropCoords = imsup.DetermineCropCoords(img2.height, img2.width, shift)
    commonCoords = imsup.GetCommonArea(commonCoords, cropCoords)
    squareCoords = imsup.MakeSquareCoords(commonCoords)
    img1Aligned = imsup.CropImageROICoords(img1, squareCoords)
    img2Aligned = imsup.CropImageROICoords(img2Shifted, squareCoords)
    img2Aligned.defocus = img1Aligned.defocus + mcfBest.defocus
    img2Aligned.shift = shift
    return img1Aligned, img2Aligned

# -------------------------------------------------------------------

def DetermineAbsoluteDefocus(imgList, idxInFocus):
    # images in imgList have relative defocus values assigned
    dfSteps = [ img.defocus for img in imgList[1:] ]
    for idx in range(len(imgList)):
        df = 0.0
        if idx < idxInFocus:
            for j in range(idx, idxInFocus):
                df -= dfSteps[j]
        elif idx > idxInFocus:
            for j in range(idxInFocus, idx):
                df += dfSteps[j]
        imgList[idx].defocus = df

# -------------------------------------------------------------------

# dorobic funkcje MoveUp, MoveDown, MoveLeft, MoveRight

def MoveImageUp(img, pxShift):
    ShiftImageAmpBuffer(img, (-pxShift, 0))

def MoveImageDown(img, pxShift):
    ShiftImageAmpBuffer(img, (pxShift, 0))

def MoveImageLeft(img, pxShift):
    ShiftImageAmpBuffer(img, (0, -pxShift))

def MoveImageRight(img, pxShift):
    ShiftImageAmpBuffer(img, (0, pxShift))