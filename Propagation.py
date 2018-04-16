import numpy as np
from numba import cuda
import math
import Constants as const
import CudaConfig as ccfg
import ArraySupport as arrsup
import ImageSupport as imsup
import CrossCorr as cc

# -------------------------------------------------------------------

def CalcTransferFunction(imgDim, pxDim, defocusChange):
    blockDim, gridDim = ccfg.DetermineCudaConfig(imgDim)
    ctf = imsup.Image(imgDim, imgDim, imsup.Image.cmp['CAP'], imsup.Image.mem['GPU'])
    ctfCoeff = np.pi * const.ewfLambda * defocusChange
    CalcTransferFunction_dev[gridDim, blockDim](ctf.amPh.am, ctf.amPh.ph, imgDim, pxDim, ctfCoeff)
    ctf.defocus = defocusChange
    return ctf

# -------------------------------------------------------------------

@cuda.jit('void(float32[:, :], int32, int32)')
def CalcRecSquareDistances_dev(rsd, imgDim, pxDim):
    x, y = cuda.grid(2)
    if x >= imgDim or y >= imgDim:
        return
    recPxWidth = 1.0 / (imgDim * pxDim)
    recOrigin = -1.0 / (2.0 * pxDim)
    recXDist = recOrigin + x * recPxWidth
    recYDist = recOrigin + y * recPxWidth
    rsd[x, y] = recXDist * recXDist + recYDist * recYDist

# -------------------------------------------------------------------

@cuda.jit('void(float32[:, :], float32[:, :], int32, float32, float32)')
def CalcTransferFunction_dev(ctfAm, ctfPh, imgDim, pxDim, ctfCoeff):
    x, y = cuda.grid(2)
    if x >= imgDim or y >= imgDim:
        return

    recPxWidth = 1.0 / (imgDim * pxDim)
    recOrigin = -1.0 / (2.0 * pxDim)

    recXDist = recOrigin + x * recPxWidth
    recYDist = recOrigin + y * recPxWidth
    recSquareDist = recXDist * recXDist + recYDist * recYDist

    ctfAm[x, y] = 1.0
    ctfPh[x, y] = ctfCoeff * recSquareDist

# -------------------------------------------------------------------

def InsertAperture(img, ap_radius):
    img_dim = img.amPh.am.shape[0]
    blockDim, gridDim = ccfg.DetermineCudaConfig(img_dim)
    img.MoveToGPU()
    img_with_aperature = imsup.CopyImage(img)
    InsertAperture_dev[gridDim, blockDim](img_with_aperature.amPh.am, img_with_aperature.amPh.ph, img_dim, ap_radius)
    return img_with_aperature

# -------------------------------------------------------------------

@cuda.jit('void(float32[:, :], float32[:, :], int32, int32)')
def InsertAperture_dev(img_am, img_ph, img_dim, ap_radius):
    x, y = cuda.grid(2)
    if x >= img_dim or y >= img_dim:
        return

    mid = img_dim // 2
    if (x - mid) ** 2 + (y - mid) ** 2 > ap_radius ** 2:
        img_am[y, x] = 0.0
        img_ph[y, x] = 0.0

# -------------------------------------------------------------------

def PropagateWave(img, ctf):
    fft = cc.FFT(img)
    fft.ReIm2AmPh()

    ctf = cc.Diff2FFT(ctf)     # !!!
    ctf.ReIm2AmPh()

    fftProp = imsup.Image(img.height, img.width, imsup.Image.cmp['CAP'], imsup.Image.mem['GPU'])
    fftProp.amPh = imsup.MultAmPhMatrices(fft.amPh, ctf.amPh)

    imgProp = cc.IFFT(fftProp)
    imgProp.ReIm2AmPh()
    imgProp.defocus = img.defocus + ctf.defocus
    return imgProp

# -------------------------------------------------------------------

def PropagateToFocus(img):
    ctf = CalcTransferFunction(img.width, img.pxWidth, -img.defocus)
    return PropagateWave(img, ctf)

# -------------------------------------------------------------------

def PropagateBackToDefocus(img, defocus):
    ctf = CalcTransferFunction(img.width, img.pxWidth, defocus)
    # img.defocus = 0.0
    return PropagateWave(img, ctf)

# -------------------------------------------------------------------

def PerformIWFR(images, N):
    imgHeight, imgWidth = images[0].height, images[0].width
    ewfResultsDir = 'results/ewf/'
    ewfAmName = 'am'
    ewfPhName = 'ph'
    backCTFunctions = []
    forwCTFunctions = []

    print('Starting IWFR...')

    # storing contrast transfer functions

    for img in images:
        # print(imgWidth, img.pxWidth, -img.defocus)
        ctf = CalcTransferFunction(imgWidth, img.pxWidth, -img.defocus)
        ctf.AmPh2ReIm()
        # ctf.reIm = ccc.Diff2FFT(ctf.reIm)
        backCTFunctions.append(ctf)

        ctf = CalcTransferFunction(imgWidth, img.pxWidth, img.defocus)
        ctf.AmPh2ReIm()
        # ctf.reIm = ccc.Diff2FFT(ctf.reIm)
        forwCTFunctions.append(ctf)

    # start IWFR procedure

    exitWave = imsup.Image(imgHeight, imgWidth, imsup.Image.cmp['CRI'], imsup.Image.mem['GPU'])

    for i in range(0, N):
        print('Iteration no {0}...'.format(i+1))
        imsup.ClearImageData(exitWave)

        # backpropagation (to in-focus plane)

        for img, idx in zip(images, range(0, len(images))):
            img = PropagateWave(img, backCTFunctions[idx])
            # img = PropagateToFocus(img)
            img.AmPh2ReIm()
            exitWave.reIm = arrsup.AddTwoArrays(exitWave.reIm, img.reIm)

        exitWave.reIm = arrsup.MultArrayByScalar(exitWave.reIm, 1/len(images))

        ewfAmPath = ewfResultsDir + ewfAmName + str(i+1) + '.png'
        ewfPhPath = ewfAmPath.replace(ewfAmName, ewfPhName)
        ewfAmplPhPath = ewfPhPath.replace(ewfPhName, ewfPhName + 'Ampl')
        imsup.SaveAmpImage(exitWave, ewfAmPath)
        imsup.SavePhaseImage(exitWave, ewfPhPath)
        amplifExitPhase = FilterPhase(exitWave)
        imsup.SavePhaseImage(amplifExitPhase, ewfAmplPhPath)

        # forward propagation (to the original focus plane)

        for img, idx in zip(images, range(0, len(images))):
            images[idx] = PropagateWave(exitWave, forwCTFunctions[idx])
            # images[idx] = PropagateBackToDefocus(exitWave, img.defocus)
            images[idx].amPh.am = img.amPh.am           # restore original amplitude

    print('All done')

    return exitWave

# -------------------------------------------------------------------

# mozna to wszystko uproscic;
# ale to filtrowanie (artefaktow) nie jest wlasciwie potrzebne
def FilterImage(img, var, stdFactor=1.0):
    if var == 'am':
        imgFiltered = FilterAmplitude(img, stdFactor)
    else:
        imgFiltered = FilterPhase(img, stdFactor)
    return imgFiltered

# -------------------------------------------------------------------

def FilterAmplitude(img, stdFactor=1.0):
    mt = img.memType
    img.MoveToCPU()
    amplifAmpImg = imsup.CopyImage(img)
    amplifAmpImg.ReIm2AmPh()
    amTemp = np.copy(amplifAmpImg.amPh.am)
    amAvg = np.average(amTemp)
    amStd = np.std(amTemp)
    amMin = amAvg - stdFactor * amStd
    amMax = amAvg + stdFactor * amStd
    amTemp2 = amTemp * (amTemp < amMax) + amMax * (amTemp > amMax)
    amplifAmpImg.amPh.am = amTemp2 * (amTemp2 > amMin) + amMin * (amTemp2 < amMin)
    amplifAmpImg.UpdateBuffer()     # !!!
    img.ChangeMemoryType(mt)
    return amplifAmpImg

# -------------------------------------------------------------------

def FilterPhase(img, stdFactor=1.0):
    mt = img.memType
    img.MoveToCPU()
    amplifPhaseImg = imsup.CopyImage(img)
    amplifPhaseImg.ReIm2AmPh()
    phTemp = np.copy(amplifPhaseImg.amPh.ph)
    phAvg = np.average(phTemp)
    phStd = np.std(phTemp)
    phMin = phAvg - stdFactor * phStd
    phMax = phAvg + stdFactor * phStd
    phTemp2 = phTemp * (phTemp < phMax) + phMax * (phTemp > phMax)
    amplifPhaseImg.amPh.ph = phTemp2 * (phTemp2 > phMin) + phMin * (phTemp2 < phMin)
    img.ChangeMemoryType(mt)
    return amplifPhaseImg
