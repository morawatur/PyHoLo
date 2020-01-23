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

#---------------------------------------`----------------------------

def rec_holo_no_ref_2(holo_fft, shift, ap_sz=const.aperture, N_hann=const.hann_win):
    sband_mid_img = imsup.ShiftImage(holo_fft, shift)
    sband_img_ap = insert_aperture(sband_mid_img, ap_sz)
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
    phs_sum = imsup.ImageExp(img1.height, img1.width)
    phs_sum.amPh.am = img1.amPh.am * img2.amPh.am
    phs_sum.amPh.ph = img1.amPh.ph + img2.amPh.ph
    return phs_sum

#-------------------------------------------------------------------

def calc_phase_diff(img1, img2):
    phs_diff = imsup.ImageExp(img1.height, img1.width)
    phs_diff.amPh.am = img1.amPh.am * img2.amPh.am
    phs_diff.amPh.ph = img1.amPh.ph - img2.amPh.ph
    return phs_diff
