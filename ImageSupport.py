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

def radians(angle):
    return angle * np.pi / 180

# -------------------------------------------------------------------

def degrees(angle):
    return angle * 180 / np.pi

#-------------------------------------------------------------------

def frange(x, y, jump):
    while x < y:
        yield x
        x += jump

#-------------------------------------------------------------------

class ComplexAmPhMatrix:
    def __init__(self, height, width):
        self.am = np.zeros((height, width), dtype=np.float32)
        self.ph = np.zeros((height, width), dtype=np.float32)

    def __del__(self):
        del self.am
        del self.ph

#-------------------------------------------------------------------

def conj_amph_matrix(ap):
    ap_conj = ComplexAmPhMatrix(ap.am.shape[0], ap.am.shape[1])
    ap_conj.am = np.copy(ap.am)
    ap_conj.ph = -ap.ph
    return ap_conj

# -------------------------------------------------------------------

def mult_amph_matrices(ap1, ap2):
    ap3 = ComplexAmPhMatrix(ap1.am.shape[0], ap1.am.shape[1])
    ap3.am = ap1.am * ap2.am
    ap3.ph = ap1.ph + ap2.ph
    return ap3

#-------------------------------------------------------------------

class Image:
    cmp = {'CRI': 0, 'CAP': 1}
    cap_var = {'AM': 0, 'PH': 1}
    cri_var = {'RE': 0, 'IM': 1}
    px_dim_default = 1.0

    def __init__(self, height, width, cmp_repr=cmp['CAP'], defocus=0.0, num=1, px_dim_sz=-1.0):
        self.width = width
        self.height = height
        self.size = width * height
        self.name = ''
        self.reim = np.zeros((height, width), dtype=np.complex64)
        self.amph = ComplexAmPhMatrix(height, width)
        self.cmp_repr = cmp_repr
        self.defocus = defocus
        self.num_in_ser = num
        self.prev = None
        self.next = None
        self.px_dim = px_dim_sz
        # self.bias = 0.0
        # self.gain = 255.0
        # self.gamma = 1.0
        if px_dim_sz < 0:
            self.px_dim = self.px_dim_default

    def __del__(self):
        self.reim = None
        self.amph.am = None
        self.amph.ph = None

    def change_complex_repr(self, new_repr):
        if new_repr == self.cmp['CAP']:
            self.reim_to_amph()
        elif new_repr == self.cmp['CRI']:
            self.amph_to_reim()

    def reim_to_amph(self):
        if self.cmp_repr == self.cmp['CAP']:
            return
        self.amph.am = np.abs(self.reim)
        self.amph.ph = np.angle(self.reim)
        # self.amph.ph = np.arctan2(self.reim.imag, self.reim.real)
        self.cmp_repr = self.cmp['CAP']

    def amph_to_reim(self):
        if self.cmp_repr == self.cmp['CRI']:
            return
        self.reim = self.amph.am * np.exp(1j * self.amph.ph)
        self.cmp_repr = self.cmp['CRI']

    def get_num_in_series_from_prev(self):
        if self.prev is not None:
            self.num_in_ser = self.prev.num_in_ser + 1

# -------------------------------------------------------------------

class ImageExp(Image):
    def __init__(self, height, width, cmp_repr=Image.cmp['CAP'], defocus=0.0, num=1, px_dim_sz=-1.0):
        super(ImageExp, self).__init__(height, width, cmp_repr, defocus, num, px_dim_sz)
        self.parent = super(ImageExp, self)
        self.shift = [0, 0]     # [dx, dy]
        self.rot = 0
        self.cos_phase = None
        self.buffer = ComplexAmPhMatrix(height, width)

    def __del__(self):
        super(ImageExp, self).__del__()
        self.buffer.am = None
        self.buffer.ph = None
        self.cos_phase = None

    def load_amp_data(self, amp_data):
        self.amph.am = np.copy(amp_data)
        self.buffer.am = np.copy(amp_data)

    def load_phs_data(self, phs_data):
        self.amph.ph = np.copy(phs_data)
        self.buffer.ph = np.copy(phs_data)

    def update_buffer(self):
        self.buffer.am = np.copy(self.amph.am)
        self.buffer.ph = np.copy(self.amph.ph)

    def update_image_from_buffer(self):
        self.amph.am = np.copy(self.buffer.am)
        self.amph.ph = np.copy(self.buffer.ph)

    def reim_to_amph(self):
        super(ImageExp, self).reim_to_amph()
        self.update_buffer()

    def update_cos_phase(self):
        self.cos_phase = np.cos(self.amph.ph)

#-------------------------------------------------------------------

class ImageList(list):
    def __init__(self, img_list=[]):
        super(ImageList, self).__init__(img_list)
        self.update_links()

    def __delitem__(self, item):
        super(ImageList, self).__delitem__(item)
        self.update_and_restrain_links()

    def update_links(self):
        for img_prev, img_next in zip(self[:-1], self[1:]):
            img_prev.next = img_next
            img_next.prev = img_prev
            img_next.num_in_ser = img_prev.num_in_ser + 1

    def update_and_restrain_links(self):
        self[0].prev = None
        self[0].num_in_ser = 1
        self[len(self)-1].next = None
        self.update_links()

#-------------------------------------------------------------------

def amph_to_new_reim(img):
    img_ri = ImageExp(img.height, img.width, cmp_repr=Image.cmp['CRI'])
    if img.cmp_repr == Image.cmp['CRI']:
        img_ri.reim = np.copy(img.reim)
    else:
        img_ri.reim = img.amph.am * np.exp(1j * img.amph.ph)
    return img_ri

# ---------------------------------------------------------------

def reim_to_new_amph(img):
    img_ap = ImageExp(img.height, img.width, cmp_repr=Image.cmp['CAP'])
    if img.cmp_repr == Image.cmp['CAP']:
        img_ap.amph.am = np.copy(img.amph.am)
        img_ap.amph.ph = np.copy(img.amph.ph)
    else:
        img_ap.amph.am = np.abs(img.reim)
        img_ap.amph.ph = np.angle(img.reim)
    return img_ap

#-------------------------------------------------------------------

def clear_image_data(img):
    shape = img.reim.shape
    img.reim = np.zeros(shape, dtype=np.complex64)
    img.amph.am = np.zeros(shape, dtype=np.float32)
    img.amph.ph = np.zeros(shape, dtype=np.float32)

#-------------------------------------------------------------------

def scale_image(img, new_min, new_max):
    curr_min = img.min()
    curr_max = img.max()
    if curr_min == curr_max:
        return img
    img_scaled = (img - curr_min) * (new_max - new_min) / (curr_max - curr_min) + new_min
    return img_scaled

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

def copy_reim_image(img):
    img_copy = ImageExp(img.height, img.width, cmp_repr=img.cmp_repr, defocus=img.defocus, num=img.num_in_ser,
                        px_dim_sz=img.px_dim)
    img_copy.reim = np.copy(img.amph.reim)
    return img_copy

#-------------------------------------------------------------------

def copy_amph_image(img):
    img_copy = ImageExp(img.height, img.width, cmp_repr=img.cmp_repr, defocus=img.defocus, num=img.num_in_ser,
                        px_dim_sz=img.px_dim)
    img_copy.amph.am = np.copy(img.amph.am)
    img_copy.amph.ph = np.copy(img.amph.ph)
    return img_copy

#-------------------------------------------------------------------

def copy_image(img):
    if img.cmp_repr == Image.cmp['CRI']:
        img_copy = copy_reim_image(img)
    else:
        img_copy = copy_amph_image(img)
    return img_copy

# -------------------------------------------------------------------

def crop_amph_roi(img, coords):
    x1, y1, x2, y2 = coords
    roi_w = x2 - x1
    roi_h = y2 - y1
    roi = ImageExp(roi_h, roi_w, img.cmp_repr, px_dim_sz=img.px_dim)

    roi.amph.am[:] = img.amph.am[y1:y2, x1:x2]
    roi.amph.ph[:] = img.amph.ph[y1:y2, x1:x2]
    roi.update_buffer()

    if img.cos_phase is not None:
        roi.update_cos_phase()
    return roi

# -------------------------------------------------------------------

def crop_img_roi_tl(img, tl_pt, roi_dims):
    tl_x, tl_y = tl_pt
    roi_w, roi_h = roi_dims
    dt = img.cmp_repr
    img.amph_to_reim()
    roi = ImageExp(roi_h, roi_w, img.cmp_repr, px_dim_sz=img.px_dim)
    roi.reim[:] = img.reim[tl_y:tl_y + roi_h, tl_x:tl_x + roi_w]
    img.change_complex_repr(dt)
    roi.change_complex_repr(dt)
    return roi

# -------------------------------------------------------------------

def crop_img_roi_mid(img, mid_pt, roi_dims):
    mid_x, mid_y = mid_pt
    roi_w, roi_h = roi_dims
    tl_pt = [mid_x - roi_w // 2, mid_y - roi_h // 2]
    roi = crop_img_roi_tl(img, tl_pt, roi_dims)
    return roi

# -------------------------------------------------------------------

def det_crop_coords(width, height, shift):
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

def det_crop_coords_after_rotation(img_dim, rot_dim, angle):
    angle_rad = radians(angle % 90)
    new_dim = int((1 / np.sqrt(2)) * img_dim / np.cos(angle_rad - radians(45)))
    orig_x = int(rot_dim / 2 - new_dim / 2)
    crop_coords = [orig_x] * 2 + [orig_x + new_dim] * 2
    return crop_coords

#-------------------------------------------------------------------

def det_crop_coords_for_new_dims(old_width, old_height, new_width, new_height):
    orig_x = (old_width - new_width) // 2
    orig_y = (old_height - new_height) // 2
    crop_coords = [orig_x, orig_y, orig_x + new_width, orig_y + new_height]
    return crop_coords

#-------------------------------------------------------------------

def calc_coords_from_new_center(p1, new_center):
    p2 = [ px - cx for px, cx in zip(p1, new_center) ]
    return p2

#-------------------------------------------------------------------

def get_common_area(coords1, coords2):
    coords3 = []
    coords3[0:2] = [ max(c1, c2) for c1, c2 in zip(coords1[0:2], coords2[0:2]) ]
    coords3[2:4] = [ min(c1, c2) for c1, c2 in zip(coords1[2:4], coords2[2:4]) ]
    return coords3

#-------------------------------------------------------------------

def make_square_coords(coords):
    height = coords[3] - coords[1]
    width = coords[2] - coords[0]
    half_diff = abs(height - width) // 2
    dim_fix = 1 if (height + width) % 2 else 0

    if height > width:
        # square_coords = [coords[1] + half_diff + dim_fix, coords[0], coords[3] - half_diff, coords[2]]
        square_coords = [coords[0], coords[1] + half_diff + dim_fix, coords[2], coords[3] - half_diff]
    else:
        # square_coords = [coords[1], coords[0] + half_diff + dim_fix, coords[3], coords[2] - half_diff]
        square_coords = [coords[0] + half_diff + dim_fix, coords[1], coords[2] - half_diff, coords[3]]
    return square_coords

#-------------------------------------------------------------------

def create_imgexp_from_img(img):
    return copy_image(img)

#-------------------------------------------------------------------

def get_first_image(img):
    first = img
    while first.prev is not None:
        first = first.prev
    return first

#-------------------------------------------------------------------

def get_last_image(img):
    last = img
    while last.next is not None:
        last = last.next
    return last

#-------------------------------------------------------------------

def create_image_list_from_first_image(img):
    img_list = ImageList()
    img_list.append(img)
    while img.next is not None:
        img = img.next
        img_list.append(img)
    return img_list

#-------------------------------------------------------------------

def create_image_list_from_image(img, how_many):
    img_list = ImageList()
    img_list.append(img)
    for idx in range(how_many-1):
        img = img.next
        img_list.append(img)
    return img_list

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
    img.reim_to_amph()

    pad_img = ImageExp(p_height, p_width, img.cmp_repr, num=img.num_in_ser, px_dim_sz=img.px_dim)
    pad_img.amph.am = pad_img.amph.am.astype(img.amph.am.dtype)
    pad_img.amph.ph = pad_img.amph.ph.astype(img.amph.ph.dtype)

    pad_array(img.amph.am, pad_img.amph.am, pads, pad_value)
    pad_array(img.amph.ph, pad_img.amph.ph, pads, pad_value)

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
    img.reim_to_amph()

    pad_img = ImageExp(p_height, p_width, img.cmp_repr, num=img.num_in_ser, px_dim_sz=img.px_dim)
    pad_img.amph.am = pad_img.amph.am.astype(img.amph.am.dtype)
    pad_img.amph.ph = pad_img.amph.ph.astype(img.amph.ph.dtype)

    pad_array(img.amph.am, pad_img.amph.am, pads, pad_value)
    pad_array(img.amph.ph, pad_img.amph.ph, pads, pad_value)

    return pad_img

#-------------------------------------------------------------------

def flip_image_h(img):
    img.amph.am = np.fliplr(img.amph.am)
    img.amph.ph = np.fliplr(img.amph.ph)

#-------------------------------------------------------------------

def flip_image_v(img):
    img.amph.am = np.flipud(img.amph.am)
    img.amph.ph = np.flipud(img.amph.ph)

#-------------------------------------------------------------------

def calc_fft(img):
    dt = img.cmp_repr
    img.amph_to_reim()

    fft = ImageExp(img.height, img.width, Image.cmp['CRI'])
    fft.reim = np.fft.fft2(img.reim)

    img.change_complex_repr(dt)
    fft.change_complex_repr(dt)
    return fft

#-------------------------------------------------------------------

def calc_ifft(fft):
    dt = fft.cmp_repr
    fft.amph_to_reim()

    ifft = ImageExp(fft.height, fft.width, Image.cmp['CRI'])
    ifft.reim = np.fft.ifft2(fft.reim)
    # ifft.reim = sp.fft.ifft2(fft.reim)
    # ifft.reim = sp.fftpack.ifft2(fft.reim)        # legacy

    fft.change_complex_repr(dt)
    ifft.change_complex_repr(dt)
    return ifft

#-------------------------------------------------------------------

def fft_to_diff(fft):
    dt = fft.cmp_repr
    fft.amph_to_reim()

    diff = ImageExp(fft.height, fft.width, cmp_repr=Image.cmp['CRI'])
    mid = diff.width // 2
    diff.reim[:mid, :mid] = fft.reim[mid:, mid:]
    diff.reim[:mid, mid:] = fft.reim[mid:, :mid]
    diff.reim[mid:, :mid] = fft.reim[:mid, mid:]
    diff.reim[mid:, mid:] = fft.reim[:mid, :mid]

    fft.change_complex_repr(dt)
    diff.change_complex_repr(dt)
    return diff

#-------------------------------------------------------------------

def diff_to_fft(diff):
    return fft_to_diff(diff)

#-------------------------------------------------------------------

def calc_cross_corr_fun(img1, img2):
    fft1 = calc_fft(img1)
    fft2 = calc_fft(img2)

    fft1.reim_to_amph()
    fft2.reim_to_amph()

    fft3 = ImageExp(fft1.height, fft1.width, Image.cmp['CAP'])
    fft1.amph = conj_amph_matrix(fft1.amph)
    fft3.amph = mult_amph_matrices(fft1.amph, fft2.amph)
    fft3.amph.am = np.sqrt(fft3.amph.am)			# mcf ON

    # ---- ccf ----
    # fft3.amph.am = fft1.amph.am * fft2.amph.am
    # fft3.amph.ph = -fft1.amph.ph + fft2.amph.ph
    # ---- mcf ----
    # fft3.amph.am = np.sqrt(fft1.amph.am * fft2.amph.am)
    # fft3.amph.ph = -fft1.amph.ph + fft2.amph.ph

    ccf = calc_ifft(fft3)
    ccf = fft_to_diff(ccf)
    ccf.reim_to_amph()
    return ccf

#-------------------------------------------------------------------

def get_shift(ccf):
    dt = ccf.cmp_repr
    ccf.reim_to_amph()

    ccf_mid_xy = np.array(ccf.amph.am.shape) // 2
    ccf_max_xy = np.array(np.unravel_index(np.argmax(ccf.amph.am), ccf.amph.am.shape))
    shift = tuple(ccf_mid_xy - ccf_max_xy)[::-1]

    ccf.change_complex_repr(dt)
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

def shift_amph_image(img, shift):
    dx, dy = shift
    img_shifted = ImageExp(img.height, img.width, img.cmp_repr, px_dim_sz=img.px_dim)

    img_shifted.amph.am = shift_array(img.amph.am, dx, dy)
    img_shifted.amph.ph = shift_array(img.amph.ph, dx, dy)

    if img.cos_phase is not None:
        img_shifted.update_cos_phase()

    return img_shifted

#-------------------------------------------------------------------

def shift_image(img, shift):
    dt = img.cmp_repr
    img.amph_to_reim()
    dx, dy = shift

    img_shifted = ImageExp(img.height, img.width, img.cmp_repr, px_dim_sz=img.px_dim)
    img_shifted.reim = shift_array(img.reim, dx, dy)

    img.change_complex_repr(dt)
    img_shifted.change_complex_repr(dt)
    return img_shifted

#-------------------------------------------------------------------

def remove_outlier_pixels(arr, min_threshold=0.2, max_threshold=1.8):
    arr_mean = np.mean(arr)
    arr_corr = np.copy(arr)
    arr_corr[arr > max_threshold * arr_mean] = arr_mean
    arr_corr[arr < min_threshold * arr_mean] = arr_mean
    return arr_corr

#-------------------------------------------------------------------

def remove_outlier_pixels_ip(arr, min_threshold=0.2, max_threshold=1.8):
    arr_mean = np.mean(arr)
    arr[arr > max_threshold * arr_mean] = arr_mean
    arr[arr < min_threshold * arr_mean] = arr_mean

#-------------------------------------------------------------------

def prep_img_and_calc_log(arr):
    arr_corr = np.copy(arr)
    arr_corr[arr <= 0] = np.min(arr[arr > 0])
    return np.log(arr_corr)

#-------------------------------------------------------------------

def prep_img_and_calc_log_ip(arr):
    arr[arr <= 0] = np.min(arr[arr > 0])
    arr[:] = np.log(arr)

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
    img_scaled = scale_image(img_src, Imin, Imax)
    # correct brightness
    img_scaled += brg
    img_scaled[img_scaled < 0.0] = 0.0
    # correct gamma
    img_scaled **= gam
    img_scaled[img_scaled > 255.0] = 255.0

    # # option 2 (c->g->b)
    # # correct contrast
    # img_scaled = scale_image(img_src, Imin, Imax)
    # img_scaled[img_scaled < 0.0] = 0.0
    # # correct gamma
    # img_scaled **= gam
    # # correct brightness
    # img_scaled += brg
    # img_scaled[img_scaled < 0.0] = 0.0
    # img_scaled[img_scaled > 255.0] = 255.0
    return img_scaled

#-------------------------------------------------------------------

def save_arr_as_tiff(data, fname, log=False, color=False, brg=0, cnt=255, gam=1.0, rm_out_px=False, min_px=0.2, max_px=1.8):
    if rm_out_px:
        remove_outlier_pixels_ip(data, min_px, max_px)
    if log:
        prep_img_and_calc_log_ip(data)
    data = update_image_bright_cont_gamma(data, brg, cnt, gam)
    if color:
        data = grayscale_to_rgb(data)
        data_to_save = im.fromarray(data.astype(np.uint8), 'RGB')
    else:
        data_to_save = im.fromarray(data.astype(np.uint8))
    data_to_save.save('{0}.tif'.format(fname))

#-------------------------------------------------------------------

def prepare_image_to_display(img, cap_var, scale=True, log=False, color=False):
    dt = img.cmp_repr
    img.reim_to_amph()
    img_var = img.amph.am if cap_var == Image.cap_var['AM'] else img.amph.ph
    img.change_complex_repr(dt)

    if log:
        if np.min(img_var) < 0:
            img_var -= np.min(img_var)
        if np.min(img_var) == 0:
            img_var += 1e-6
        img_var = np.log10(img_var)

    if scale:
        img_var = scale_image(img_var, 0.0, 255.0)

    if not color:
        img_to_disp = im.fromarray(img_var.astype(np.uint8))
    else:
        img_var_col = grayscale_to_rgb(img_var)
        img_to_disp = im.fromarray(img_var_col.astype(np.uint8), 'RGB')

    return img_to_disp

#-------------------------------------------------------------------

def display_amp_image(img, scale=True, log=False):
    img_to_disp = prepare_image_to_display(img, Image.cap_var['AM'], scale, log)
    img_to_disp.show()

# -------------------------------------------------------------------

def save_amp_image(img, fpath, scale=True, log=False, color=False):
    img_to_save = prepare_image_to_display(img, Image.cap_var['AM'], scale, log, color)
    img_to_save.save(fpath)

#-------------------------------------------------------------------

def display_phase_image(img, scale=True, log=False):
    img_to_disp = prepare_image_to_display(img, Image.cap_var['PH'], scale, log)
    img_to_disp.show()

# -------------------------------------------------------------------

def save_phase_image(img, fpath, scale=True, log=False, color=False):
    img_to_save = prepare_image_to_display(img, Image.cap_var['PH'], scale, log, color)
    img_to_save.save(fpath)

#-------------------------------------------------------------------

def fill_image_with_value(img, value):
    img.reim_to_amph()
    img.amph.am.fill(value)