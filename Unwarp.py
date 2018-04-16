import numpy as np
import ImageSupport as imsup
import CrossCorr as cc
import Propagation as prop
from scipy import interpolate
from scipy import ndimage
from skimage import transform as tf

# -------------------------------------------------------------------

# zakladam, ze obrazy sa wstepnie zsuniete
def UnwarpImage(imgRef, img, nDiv, fragCoords):
    mt = img.memType
    dt = img.cmpRepr

    dfChange = -abs(abs(imgRef.defocus) - abs(img.defocus))
    print('df_uw({0}, {1}) = {2:.2f} um'.format(imgRef.numInSeries, img.numInSeries, dfChange * 1e6))
    # imgRefProp = prop.PropagateBackToDefocus(imgRef, dfChange)

    # fragCoords = [(b, a) for a in range(nDiv) for b in range(nDiv)]
    shifts = cc.CalcPartialCrossCorrFunUW(imgRef, img, nDiv, fragCoords)

    # interpolacja

    shiftsY = np.copy(shifts.real)
    shiftsX = np.copy(shifts.imag)
    shiftsX[shiftsX == 0] = np.nan
    shiftsY[shiftsY == 0] = np.nan

    x = np.arange(0, shifts.shape[0])
    y = np.arange(0, shifts.shape[1])
    # mask invalid values
    shiftsX = np.ma.masked_invalid(shiftsX)
    shiftsY = np.ma.masked_invalid(shiftsY)
    xx, yy = np.meshgrid(x, y)
    # get only the valid values
    x1 = xx[~shiftsX.mask]
    y1 = yy[~shiftsX.mask]
    x2 = xx[~shiftsY.mask]
    y2 = yy[~shiftsY.mask]
    newShiftsX = shiftsX[~shiftsX.mask]
    newShiftsY = shiftsY[~shiftsY.mask]

    GDX = interpolate.griddata((x1, y1), newShiftsX.ravel(), (xx, yy), method='nearest')
    GDY = interpolate.griddata((x2, y2), newShiftsY.ravel(), (xx, yy), method='nearest')
    # print(type(GDX[0, 0]))
    newShifts = [(int(gdx), int(gdy)) for gdx, gdy in zip(GDX.reshape(nDiv ** 2), GDY.reshape(nDiv ** 2))]
    # oldShifts = [(int(ox), int(oy)) for ox, oy in zip(shifts.reshape(nDiv ** 2).imag, shifts.reshape(nDiv ** 2).real)]
    # print(oldShifts)
    # newShifts = GDX + 1j * GDY

    # ---
    allFragCoords = [(b, a) for a in range(nDiv) for b in range(nDiv)]
    roiNR, roiNC = img.height // nDiv, img.width // nDiv
    fragsToCorrelate1 = []
    fragsToCorrelate2 = []

    for x, y in allFragCoords:
        frag1 = imsup.CropImageROI(imgRef, (y * roiNR, x * roiNC), (roiNR, roiNC), 1)
        fragsToCorrelate1.append(frag1)

        frag2 = imsup.CropImageROI(img, (y * roiNR, x * roiNC), (roiNR, roiNC), 1)
        fragsToCorrelate2.append(frag2)

    fragsToJoin = imsup.ImageList()
    for shift, frag1, frag2 in zip(newShifts, fragsToCorrelate1, fragsToCorrelate2):
        frag2Shifted = cc.ShiftImage(frag2, shift)
        fragsToJoin.append(frag2Shifted)

    img2Unwarped = imsup.JoinImages(fragsToJoin, nDiv)
    imsup.SaveAmpImage(img2Unwarped, 'warp_field.png')
    # ---

    # allFragCoords = [(b, a) for a in range(nDiv) for b in range(nDiv)]
    fragDimSize = img.width // nDiv
    src = np.array(allFragCoords)
    print(src, type(src))
    src *= fragDimSize
    dst = src - newShifts

    img.ReIm2AmPh()
    img.MoveToCPU()
    oldMin, oldMax = np.min(img.amPh.am), np.max(img.amPh.am)
    scaledArray = imsup.ScaleImage(img.amPh.am, -1.0, 1.0)

    tform3 = tf.ProjectiveTransform()
    tform3.estimate(src, dst)
    warped = tf.warp(scaledArray, tform3, output_shape=(img.height, img.width)).astype(np.float32)
    warpedScaledBack = imsup.ScaleImage(warped, oldMin, oldMax)

    warpedImage = imsup.Image(warped.shape[0], warped.shape[1])
    warpedImage.amPh.am = np.copy(warpedScaledBack)
    # img.amPh.am = np.copy(warpedImage.amPh.am)

    img.ChangeMemoryType(mt)
    img.ChangeComplexRepr(dt)
    imgRef.ChangeMemoryType(mt)
    imgRef.ChangeComplexRepr(dt)

    # imsup.SaveAmpImage(imgRef, 'ref.png')
    # imsup.SaveAmpImage(img, 'uwImg.png')

    return warpedImage

# -------------------------------------------------------------------

def UnwarpImageList(imgList, nDiv):
    # imgListCopy = imsup.ImageList()
    # for img in imgList:
    #    imgListCopy.append(imsup.CopyImage(img))
    # for img, imgRef, idx in zip(imgList[1:], imgListCopy[:len(imgListCopy) - 1], range(1, len(imgList))):

    # ---
    imgRef = imgList[0]
    imgRef2 = imsup.CopyImage(imgList[10])
    for img, idx in zip(imgList[1:11], range(1, 11)):
        df = img.defocus
        imgList[idx] = UnwarpImage(imgRef, img, nDiv)
        imgList[idx].defocus = df
        warpPath = 'results/warp/imgw{0}.png'.format(img.numInSeries)
        imsup.SaveAmpImage(imgList[idx], warpPath)

    for img, idx in zip(imgList[11:], range(11, len(imgList))):
        df = img.defocus
        imgList[idx] = UnwarpImage(imgRef2, img, nDiv)
        imgList[idx].defocus = df
        warpPath = 'results/warp/imgw{0}.png'.format(img.numInSeries)
        imsup.SaveAmpImage(imgList[idx], warpPath)
    # ---

    # imgList.reverse()
    # imgRef = imgList[0]
    # for img, idx in zip(imgList[1:], range(1, len(imgList))):
    #     df = img.defocus
    #     imgList[idx] = UnwarpImage(imgRef, img, nDiv)
    #     imgList[idx].defocus = df
    #     warpPath = 'results/warpNew/imgw{0}.png'.format(img.numInSeries)
    #     imsup.SaveAmpImage(imgList[idx], warpPath)
    # imgList.reverse()
    return imgList
