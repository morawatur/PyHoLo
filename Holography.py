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
    max_xy = list(np.unravel_index(np.argmax(img), img.shape))[::-1]
    # max_xy.reverse()      # reverse() is the same as [::-1]
    return max_xy

#-------------------------------------------------------------------

def insert_aperture(img, ap):
    dt = img.cmp_repr
    img.reim_to_amph()
    img_ap = imsup.copy_amph_image(img)

    n = img_ap.width
    c = n // 2
    y, x = np.ogrid[-c:n - c, -c:n - c]
    mask = x * x + y * y > ap * ap

    img_ap.amph.am[mask] = 0.0
    img_ap.amph.ph[mask] = 0.0

    img.change_complex_repr(dt)
    img_ap.change_complex_repr(dt)
    return img_ap

# -------------------------------------------------------------------

def mult_by_hann_window(img, N=100):
    dt = img.cmp_repr
    img.reim_to_amph()
    new_img = imsup.copy_amph_image(img)

    hann = np.hanning(N)
    hann_2d = np.sqrt(np.outer(hann, hann))

    hann_win = imsup.ImageExp(N, N, cmp_repr=imsup.Image.cmp['CAP'])
    hann_win.load_amp_data(hann_2d)

    hmin, hmax = (img.width - N) // 2, (img.width + N) // 2
    new_img.amph.am[hmin:hmax, hmin:hmax] *= hann_2d
    new_img.amph.ph[hmin:hmax, hmin:hmax] *= hann_2d

    img.change_complex_repr(dt)
    new_img.change_complex_repr(dt)
    return new_img

#-------------------------------------------------------------------

def holo_fft(h_img):
    h_fft = imsup.calc_fft(h_img)
    h_fft = imsup.fft_to_diff(h_fft)
    return h_fft

#-------------------------------------------------------------------

def holo_get_sideband(h_fft, shift, ap_rad=const.aperture, N_hann=const.hann_win):
    subpx_shift, px_shift = np.modf(shift)
    sband_ctr = imsup.shift_image(h_fft, list(px_shift.astype(np.int32)))
    if abs(subpx_shift[0]) > 0.0 or abs(subpx_shift[1]) > 0.0:
        sband_ctr = subpixel_shift(sband_ctr, list(subpx_shift))
    sband_ctr_ap = insert_aperture(sband_ctr, ap_rad)
    sband_ctr_ap = mult_by_hann_window(sband_ctr_ap, N=N_hann)
    return sband_ctr_ap

#-------------------------------------------------------------------

def holo_ifft(h_fft):
    h_fft = imsup.diff_to_fft(h_fft)
    h_ifft = imsup.calc_ifft(h_fft)
    return h_ifft

#-------------------------------------------------------------------

def calc_phase_sum(img1, img2):
    # complex multiplication
    phs_sum = imsup.ImageExp(img1.height, img1.width)
    phs_sum.amph.am = img1.amph.am * img2.amph.am
    phs_sum.amph.ph = img1.amph.ph + img2.amph.ph
    return phs_sum

#-------------------------------------------------------------------

def calc_phase_diff(img1, img2):
    # complex division
    phs_diff = imsup.ImageExp(img1.height, img1.width)
    img1.amph.am[np.where(img1.amph.am == 0.0)] = 1.0
    phs_diff.amph.am = img2.amph.am / img1.amph.am
    phs_diff.amph.ph = img2.amph.ph - img1.amph.ph
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
    return xm, ym

#-------------------------------------------------------------------

def find_sideband_center(sband, orig=(0, 0), subpx=False):
    # general convention is (x, y), i.e. (col, row)

    sb_xy = find_img_max(sband)
    sband_xy = list(np.array(sb_xy) + np.array(orig))

    if subpx:
        print('Sband precentered at (x, y) = ({0}, {1})'.format(orig[0] + sb_xy[0], orig[1] + sb_xy[1]))
        roi_hlf_edge = min([const.min_COM_roi_hlf_edge] + sb_xy + list(np.array(sband.shape[::-1]) - np.array(sb_xy)))
        print('Edge of center of mass ROI = {0}'.format(2 * roi_hlf_edge))
        sb_x1, sb_x2 = sb_xy[0] - roi_hlf_edge, sb_xy[0] + roi_hlf_edge
        sb_y1, sb_y2 = sb_xy[1] - roi_hlf_edge, sb_xy[1] + roi_hlf_edge
        sband_precentered = np.copy(sband[sb_y1:sb_y2, sb_x1:sb_x2])
        sband_xy = list(find_mass_center(sband_precentered))
        sband_xy = list(np.array(sband_xy) + np.array([sb_x1, sb_y1]) + np.array(orig))

    print('Sband found at (x, y) = ({0:.2f}, {1:.2f})'.format(sband_xy[0], sband_xy[1]))

    return sband_xy

#-------------------------------------------------------------------

def subpixel_shift(img, subpx_sh):
    dt = img.cmp_repr
    img.amph_to_reim()
    h, w = img.reim.shape
    img_sh = imsup.ImageExp(h, w, img.cmp_repr)

    dx, dy = subpx_sh
    wx = np.abs(dx)
    wy = np.abs(dy)
    arr = np.copy(img.reim)

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
    img_sh.reim = np.copy(arr_shy)

    img.change_complex_repr(dt)
    img_sh.change_complex_repr(dt)
    return img_sh

#-------------------------------------------------------------------

# def subpixel_shift(img, subpx_sh):
#     dt = img.cmp_repr
#     img.reim_to_amph()
#     h, w = img.amph.am.shape
#     img_sh = imsup.ImageExp(h, w, img.cmp_repr)
#
#     dx, dy = subpx_sh
#     wx = np.abs(dx)
#     wy = np.abs(dy)
#     amp = np.copy(img.amph.am)
#     phs = np.copy(img.amph.ph)
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
#     img_sh.amph.am = np.copy(amp_shy)
#     img_sh.amph.ph = np.copy(phs)
#     img_sh.amph.ph = np.copy(phs_shy)
#
#     img.change_complex_repr(dt)
#     img_sh.change_complex_repr(dt)
#     return img_sh
