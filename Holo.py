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
import Constants as const
import ImageSupport as imsup

#-------------------------------------------------------------------

def mask_fft(fft, mid, r, out=True):
    if out:
        mfft = np.copy(fft)
        mfft[mid[0] - r:mid[0] + r, mid[1] - r:mid[1] + r] = 0
    else:
        mfft = np.zeros(fft.shape, dtype=fft.dtype)
        mfft[mid[0] - r:mid[0] + r, mid[1] - r:mid[1] + r] = np.copy(fft[mid[0] - r:mid[0] + r, mid[1] - r:mid[1] + r])
    return mfft

#-------------------------------------------------------------------

def mask_fft_center(fft, r, out=True):
    mid = (fft.shape[0] // 2, fft.shape[1] // 2)
    return mask_fft(fft, mid, r, out)

#-------------------------------------------------------------------

def find_img_max(img):
    max_xy = np.array(np.unravel_index(np.argmax(img), img.shape))
    return list(max_xy)

#-------------------------------------------------------------------

def insert_aperture(img, ap):
    dt = img.cmpRepr
    img.ReIm2AmPh()
    img_ap = imsup.copy_am_ph_image(img)

    n = img_ap.width
    c = n // 2
    y, x = np.ogrid[-c:n - c, -c:n - c]
    mask = x * x + y * y > ap * ap

    img_ap.amPh.am[mask] = 0.0
    img_ap.amPh.ph[mask] = 0.0

    img.ChangeComplexRepr(dt)
    img_ap.ChangeComplexRepr(dt)
    return img_ap

# -------------------------------------------------------------------

def mult_by_hann_window(img, N=100):
    dt = img.cmpRepr
    img.ReIm2AmPh()
    new_img = imsup.copy_am_ph_image(img)

    hann = np.hanning(N)
    hann_2d = np.sqrt(np.outer(hann, hann))

    hann_win = imsup.ImageExp(N, N, cmpRepr=imsup.Image.cmp['CAP'])
    hann_win.LoadAmpData(hann_2d)

    hmin, hmax = (img.width - N) // 2, (img.width + N) // 2
    new_img.amPh.am[hmin:hmax, hmin:hmax] *= hann_2d
    new_img.amPh.ph[hmin:hmax, hmin:hmax] *= hann_2d

    img.ChangeComplexRepr(dt)
    new_img.ChangeComplexRepr(dt)
    return new_img

#-------------------------------------------------------------------

def rec_holo_no_ref_1(holo_img):
    holo_img.AmPh2ReIm()
    holo_fft = imsup.FFT(holo_img)
    holo_fft = imsup.FFT2Diff(holo_fft)
    holo_img.ReIm2AmPh()
    holo_fft.ReIm2AmPh()
    return holo_fft

#-------------------------------------------------------------------

def rec_holo_no_ref_2(holo_fft, shift, ap_rad=const.aperture, N_hann=const.hann_win):
    subpx_shift, px_shift = np.modf(shift)
    sband_mid_img = imsup.ShiftImage(holo_fft, list(px_shift.astype(np.int32)))
    if abs(subpx_shift[0]) > 0.0 or abs(subpx_shift[1]) > 0.0:
        sband_mid_img = subpixel_shift(sband_mid_img, list(subpx_shift))
    sband_img_ap = insert_aperture(sband_mid_img, ap_rad)
    sband_img_ap = mult_by_hann_window(sband_img_ap, N=N_hann)
    return sband_img_ap

#-------------------------------------------------------------------

def rec_holo_no_ref_3(sband_img):
    sband_img = imsup.Diff2FFT(sband_img)
    rec_holo = imsup.IFFT(sband_img)
    return rec_holo

#-------------------------------------------------------------------

def rec_holo_no_ref(holo_img, rec_sz=128, ap_sz=32, mask_sz=50, N_hann=100):
    holo_fft = imsup.FFT(holo_img)
    holo_fft = imsup.FFT2Diff(holo_fft)    # diff is re_im
    holo_fft.ReIm2AmPh()

    mfft = mask_fft_center(holo_fft.amPh.am, mask_sz, True)
    sband_xy = find_img_max(mfft)

    mid = holo_img.width // 2
    shift = [ mid - sband_xy[0], mid - sband_xy[1] ]
    sband_img = imsup.ShiftImage(holo_fft, shift)

    sband_img_ap = mult_by_hann_window(sband_img, N=N_hann)
    sband_img_ap = insert_aperture(sband_img_ap, ap_sz)

    sband_img_ap = imsup.Diff2FFT(sband_img_ap)
    rec_holo = imsup.IFFT(sband_img_ap)

    # factor = holo_img.width / rec_sz
    # rec_holo_resc = tr.RescaleImageSki(rec_holo, factor)
    rec_holo = imsup.create_imgexp_from_img(rec_holo)
    return rec_holo

#-------------------------------------------------------------------

def calc_phase_sum(img1, img2):
    # complex multiplication
    phs_sum = imsup.ImageExp(img1.height, img1.width)
    phs_sum.amPh.am = img1.amPh.am * img2.amPh.am
    phs_sum.amPh.ph = img1.amPh.ph + img2.amPh.ph
    return phs_sum

#-------------------------------------------------------------------

def calc_phase_diff(img1, img2):
    # complex division
    phs_diff = imsup.ImageExp(img1.height, img1.width)
    img1.amPh.am[np.where(img1.amPh.am == 0.0)] = 1.0
    phs_diff.amPh.am = img2.amPh.am / img1.amPh.am
    phs_diff.amPh.ph = img2.amPh.ph - img1.amPh.ph
    return phs_diff

#-------------------------------------------------------------------

def find_mass_center(arr):
    h, w = arr.shape
    x = np.linspace(0, w-1, w)
    y = np.linspace(0, h-1, h)
    xv, yv = np.meshgrid(x, y)
    all_sum = np.sum(arr)
    xm = np.sum(arr * xv) / all_sum
    ym = np.sum(arr * yv) / all_sum
    return ym, xm

#-------------------------------------------------------------------

def find_sideband_center(sband, orig=(0, 0), subpx=False):
    # general convention is (y, x), i.e. (r, c)

    sb_xy = find_img_max(sband)
    sband_xy = list(np.array(sb_xy) + np.array(orig))

    if subpx:
        print('Sband precentered at (x, y) = ({0}, {1})'.format(orig[1] + sb_xy[1], orig[0] + sb_xy[0]))
        roi_hlf_edge = min([const.min_COM_roi_hlf_edge] + sb_xy + list(np.array(sband.shape) - np.array(sb_xy)))
        print('Edge of center of mass ROI = {0}'.format(2 * roi_hlf_edge))
        sb_y1, sb_y2 = sb_xy[0] - roi_hlf_edge, sb_xy[0] + roi_hlf_edge
        sb_x1, sb_x2 = sb_xy[1] - roi_hlf_edge, sb_xy[1] + roi_hlf_edge
        sband_precentered = np.copy(sband[sb_y1:sb_y2, sb_x1:sb_x2])
        sband_xy = list(find_mass_center(sband_precentered))
        sband_xy = list(np.array(sband_xy) + np.array([sb_y1, sb_x1]) + np.array(orig))

    print('Sband found at (x, y) = ({0:.2f}, {1:.2f})'.format(sband_xy[1], sband_xy[0]))

    return sband_xy

#-------------------------------------------------------------------

def subpixel_shift(img, subpx_sh):
    dt = img.cmpRepr
    img.AmPh2ReIm()
    h, w = img.reIm.shape
    img_sh = imsup.ImageExp(h, w, img.cmpRepr)

    dy, dx = subpx_sh
    wx = np.abs(dx)
    wy = np.abs(dy)
    arr = np.copy(img.reIm)

    arr_shx_1px = np.zeros((h, w), dtype=np.complex64)
    arr_shy_1px = np.zeros((h, w), dtype=np.complex64)

    # x dir
    if dx > 0:
        arr_shx_1px[:, 1:] = arr[:, :w-1]
    else:
        arr_shx_1px[:, :w-1] = arr[:, 1:]

    arr_shx = wx * arr_shx_1px + (1.0-wx) * arr

    # y dir
    if dy > 0:
        arr_shy_1px[1:, :] = arr_shx[:h-1, :]
    else:
        arr_shy_1px[:h-1, :] = arr_shx[1:, :]

    arr_shy = wy * arr_shy_1px + (1.0-wy) * arr_shx
    img_sh.reIm = np.copy(arr_shy)

    img.ChangeComplexRepr(dt)
    img_sh.ChangeComplexRepr(dt)
    return img_sh

#-------------------------------------------------------------------

# def subpixel_shift(img, subpx_sh):
#     dt = img.cmpRepr
#     img.ReIm2AmPh()
#     h, w = img.amPh.am.shape
#     img_sh = imsup.ImageExp(h, w, img.cmpRepr)
#
#     dy, dx = subpx_sh
#     wx = np.abs(dx)
#     wy = np.abs(dy)
#     amp = np.copy(img.amPh.am)
#     phs = np.copy(img.amPh.ph)
#
#     amp_shx_1px = np.zeros((h, w), dtype=np.float32)
#     amp_shy_1px = np.zeros((h, w), dtype=np.float32)
#     phs_shx_1px = np.zeros((h, w), dtype=np.float32)
#     phs_shy_1px = np.zeros((h, w), dtype=np.float32)
#
#     # x dir
#     if dx > 0:
#         amp_shx_1px[:, 1:] = amp[:, :w-1]
#         phs_shx_1px[:, 1:] = phs[:, :w-1]
#     else:
#         amp_shx_1px[:, :w-1] = amp[:, 1:]
#         phs_shx_1px[:, :w-1] = phs[:, 1:]
#
#     amp_shx = wx * amp_shx_1px + (1.0-wx) * amp
#     phs_shx = wx * phs_shx_1px + (1.0-wx) * phs
#
#     # y dir
#     if dy > 0:
#         amp_shy_1px[1:, :] = amp_shx[:h-1, :]
#         phs_shy_1px[1:, :] = phs_shx[:h-1, :]
#     else:
#         amp_shy_1px[:h-1, :] = amp_shx[1:, :]
#         phs_shy_1px[:h-1, :] = phs_shx[1:, :]
#
#     amp_shy = wy * amp_shy_1px + (1.0-wy) * amp_shx
#     phs_shy = wy * phs_shy_1px + (1.0-wy) * phs_shx
#
#     img_sh.amPh.am = np.copy(amp_shy)
#     img_sh.amPh.ph = np.copy(phs)
#     img_sh.amPh.ph = np.copy(phs_shy)
#
#     img.ChangeComplexRepr(dt)
#     img_sh.ChangeComplexRepr(dt)
#     return img_sh
