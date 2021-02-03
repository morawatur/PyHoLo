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
import ImageSupport as imsup
from skimage import transform as tr
from skimage.restoration import unwrap_phase    # used in GUI.py

#-------------------------------------------------------------------

def rescale_pixel_dim(px_dim, old_dim, new_dim):
    resc_factor = old_dim / new_dim
    return px_dim * resc_factor

#-------------------------------------------------------------------

def rotate_image_ski(img, angle, mode='constant'):
    dt = img.cmpRepr
    img.ReIm2AmPh()

    # amp_cval = (np.min(img.amPh.am) + np.max(img.amPh.am)) / 2.0
    # phs_cval = (np.min(img.amPh.ph) + np.max(img.amPh.ph)) / 2.0

    amp_rot = tr.rotate(img.amPh.am, angle, mode=mode, cval=0.0)
    phs_rot = tr.rotate(img.amPh.ph, angle, mode=mode, cval=0.0)

    img_rot = imsup.ImageExp(amp_rot.shape[0], amp_rot.shape[1], num=img.numInSeries, px_dim_sz=img.px_dim)
    img_rot.LoadAmpData(amp_rot)
    img_rot.LoadPhsData(phs_rot)
    if img.cos_phase is not None:
        img_rot.update_cos_phase()

    img.ChangeComplexRepr(dt)
    img_rot.ChangeComplexRepr(dt)

    return img_rot

#-------------------------------------------------------------------

def rescale_image_ski(img, factor):
    dt = img.cmpRepr
    img.ReIm2AmPh()

    amp_mag = tr.rescale(img.amPh.am, scale=factor, mode='constant', cval=0.0, multichannel=False, anti_aliasing=False)
    phs_mag = tr.rescale(img.amPh.ph, scale=factor, mode='constant', cval=0.0, multichannel=False, anti_aliasing=False)

    img_mag = imsup.ImageExp(amp_mag.shape[0], amp_mag.shape[1], num=img.numInSeries, px_dim_sz=img.px_dim)
    img_mag.LoadAmpData(amp_mag)
    img_mag.LoadPhsData(phs_mag)
    if img.cos_phase is not None:
        img_mag.update_cos_phase()

    img_mag.px_dim = rescale_pixel_dim(img.px_dim, img.width, img_mag.width)

    img.ChangeComplexRepr(dt)
    img_mag.ChangeComplexRepr(dt)

    return img_mag

#-------------------------------------------------------------------

def warp_image_ski(img, src_set, dst_set):
    dt = img.cmpRepr
    img.ReIm2AmPh()

    tform3 = tr.ProjectiveTransform()
    tform3.estimate(src_set, dst_set)
    amp_warp = tr.warp(img.amPh.am, tform3, output_shape=img.amPh.am.shape, mode='constant', cval=0.0)
    phs_warp = tr.warp(img.amPh.ph, tform3, output_shape=img.amPh.ph.shape, mode='constant', cval=0.0)

    img_warp = imsup.ImageExp(amp_warp.shape[0], amp_warp.shape[1], num=img.numInSeries, px_dim_sz=img.px_dim)
    img_warp.LoadAmpData(amp_warp)
    img_warp.LoadPhsData(phs_warp)
    if img.cos_phase is not None:
        img_warp.update_cos_phase()

    img.ChangeComplexRepr(dt)
    img_warp.ChangeComplexRepr(dt)

    return img_warp

#-------------------------------------------------------------------

def det_crop_coords_after_ski_rotation(old_dim, angle):
    return imsup.det_crop_coords_after_rotation(old_dim, old_dim, angle)

#-------------------------------------------------------------------

class Line:
    def __init__(self, a_coeff, b_coeff):
        self.a = a_coeff
        self.b = b_coeff

    def get_from_2_points(self, p1, p2):
        if p1[0] != p2[0]:
            self.a = (p2[1] - p1[1]) / (p2[0] - p1[0])
            self.b = p1[1] - self.a * p1[0]
        else:
            self.a = None
            self.b = None

    def get_from_slope_and_point(self, a_coeff, p1):
        self.a = a_coeff
        self.b = p1[1] - self.a * p1[0]

# -------------------------------------------------------------------

class Plane:
    def __init__(self, a_coeff, b_coeff, c_coeff):
        self.a = a_coeff
        self.b = b_coeff
        self.c = c_coeff

    # doesn't work
    def get_from_3_points(self, p1, p2, p3):
        dx1, dy1, dz1 = list(np.array(p1) - np.array(p2))
        dx2, dy2, dz2 = list(np.array(p3) - np.array(p2))
        a = dy1 * dz2 - dy2 * dz1
        b = dz1 * dx2 - dz2 * dx1
        c = dx1 * dy2 - dx2 * dy1
        d = -(a * p2[0] + b * p2[1] + c * p2[2])
        self.a = -a / c
        self.b = -b / c
        self.c = -d / c

    def fill_plane(self, w, h):
        arr = np.zeros((h, w), dtype=np.float32)
        for y in range(h):
            for x in range(w):
                arr[y, x] = self.a * x + self.b * y + self.c
        return arr

# -------------------------------------------------------------------

def find_perpendicular_line(line, point):
    line_perp = Line(-1 / line.a, 0)
    line_perp.get_from_slope_and_point(line_perp.a, point)
    return line_perp

#-------------------------------------------------------------------

def find_rotation_center(pts1, pts2):
    A1, B1 = pts1
    A2, B2 = pts2

    Am = [np.average([A1[0], A2[0]]), np.average([A1[1], A2[1]])]
    Bm = [np.average([B1[0], B2[0]]), np.average([B1[1], B2[1]])]

    a_line = Line(0, 0)
    b_line = Line(0, 0)
    a_line.get_from_2_points(A1, A2)
    b_line.get_from_2_points(B1, B2)

    a_line_perp = find_perpendicular_line(a_line, Am)
    b_line_perp = find_perpendicular_line(b_line, Bm)

    rot_center_x = (b_line_perp.b - a_line_perp.b) / (a_line_perp.a - b_line_perp.a)
    rot_center_y = a_line_perp.a * rot_center_x + a_line_perp.b

    return [rot_center_x, rot_center_y]

#-------------------------------------------------------------------

def rotate_point(p1, angle):
    z1 = np.complex(p1[0], p1[1])
    r = np.abs(z1)
    phi = np.angle(z1) + imsup.radians(angle)
    p2 = [r * np.cos(phi), r * np.sin(phi)]
    return p2

#-------------------------------------------------------------------

def find_dir_angle(p1, p2, orig=(0, 0)):
    p1 = np.array(p1)
    p2 = np.array(p2)
    orig = np.array(orig)
    p2 += (orig - p1)
    return np.arctan2(p2[1], p2[0])

#-------------------------------------------------------------------

def find_dir_angles(p1, p2):
    lpt = p1[:] if p1[0] < p2[0] else p2[:]     # left point
    rpt = p1[:] if p1[0] > p2[0] else p2[:]     # right point
    dx = np.abs(rpt[0] - lpt[0])
    dy = np.abs(rpt[1] - lpt[1])
    sign = 1 if rpt[1] < lpt[1] else -1
    proj_dir = 1         # projection on y axis
    if dx > dy:
        sign *= -1
        proj_dir = 0     # projection on x axis
    diff1 = dx if dx < dy else dy
    diff2 = dx if dx > dy else dy
    ang1 = np.arctan2(diff1, diff2)
    ang2 = np.pi / 2 - ang1
    ang1 *= sign
    ang2 *= (-sign)
    return ang1, ang2, proj_dir

#-------------------------------------------------------------------

def calc_distance(p1, p2):
    dist = np.sqrt((p1[0] - p2[0]) ** 2 + (p1[1] - p2[1]) ** 2)
    return dist

#-------------------------------------------------------------------

def calc_inner_angle(a, b, c):
    alpha = np.arccos(np.abs((a*a + b*b - c*c) / (2*a*b)))
    return imsup.degrees(alpha)

#-------------------------------------------------------------------

def calc_outer_angle(p1, p2):
    dist = calc_distance(p1, p2)
    betha = np.arcsin(np.abs(p1[0] - p2[0]) / dist)
    return imsup.degrees(betha)

#-------------------------------------------------------------------

def calc_rot_angle(p1, p2):
    z1 = np.complex(p1[0], p1[1])
    z2 = np.complex(p2[0], p2[1])
    phi1 = np.angle(z1)
    phi2 = np.angle(z2)
    # rot_angle = np.abs(imsup.degrees(phi2 - phi1))
    rot_angle = imsup.degrees(phi2 - phi1)
    if np.abs(rot_angle) > 180:
        rot_angle = -np.sign(rot_angle) * (360 - np.abs(rot_angle))
    return rot_angle

#-------------------------------------------------------------------

def convert_points_to_tl_br(p1, p2):
    tl = list(np.amin([p1, p2], axis=0))
    br = list(np.amax([p1, p2], axis=0))
    return tl, br

#-------------------------------------------------------------------

def lin_least_squares(x_arr, y_arr):
    n_pt = len(x_arr)
    sx = np.sum(x_arr)
    sy = np.sum(y_arr)
    sxy = np.sum(np.array(x_arr) * np.array(y_arr))
    sx2 = np.sum(np.array(x_arr) ** 2)
    a_coeff = (n_pt * sxy - sx * sy) / (n_pt * sx2 - sx * sx)
    b_coeff = (sy - a_coeff * sx) / n_pt
    return a_coeff, b_coeff

#-------------------------------------------------------------------

# x_arr and y_arr must be numpy.arrays
def lin_least_squares_alt(x_arr, y_arr):
    xm, ym = np.mean(x_arr), np.mean(y_arr)
    xm_arr = x_arr - xm
    ym_arr = y_arr - ym
    sxy = np.sum(xm_arr * ym_arr)
    sx2 = np.sum(xm_arr * xm_arr)
    a_coeff = sxy / sx2
    b_coeff = ym - a_coeff * xm
    return a_coeff, b_coeff

#-------------------------------------------------------------------

def calc_avg_neigh(arr, x, y, nn=1):
    h, w = arr.shape
    x1, x2 = x - nn, x + nn + 1
    y1, y2 = y - nn, y + nn + 1
    arr_2_avg = arr[max(0, y1):min(y2, h), max(0, x1):min(x2, w)]
    n_el = arr_2_avg.size
    avg_val = np.sum(arr_2_avg) / n_el
    return avg_val

#-------------------------------------------------------------------

def calc_corr_coef(arr1, arr2):
    a1 = arr1 - np.mean(arr1)
    a2 = arr2 - np.mean(arr2)
    corr_coef = np.sum(a1 * a2) / np.sqrt(np.sum(a1 * a1) * np.sum(a2 * a2))
    return corr_coef

#-------------------------------------------------------------------

def calc_corr_array(arr1, arr2, max_shift=20):
    h, w = arr1.shape
    x1 = y1 = -max_shift
    x2 = y2 = max_shift

    roi1 = np.copy(arr1[y2:h-y2, x2:w-x2])
    corr_arr = np.zeros((2*max_shift, 2*max_shift), dtype=np.float32)

    it = 0
    n_el = corr_arr.size

    for y in range(y1, y2):
        for x in range(x1, x2):
            arr2_sh = imsup.shift_array(arr2, x, y)
            roi2 = np.copy(arr2_sh[y2:h-y2, x2:w-x2])
            corr_coef = calc_corr_coef(roi1, roi2)
            corr_arr[y2+y, x2+x] = corr_coef

            it += 1
            print('\r{0:.0f}%'.format(it*100.0/n_el), end='')

    print('\rDone!')
    return corr_arr

