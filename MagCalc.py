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
import Transform as tr

import matplotlib.pyplot as plt
from matplotlib.patches import Circle

#-------------------------------------------------------------------

def calc_B(ph1, ph2, d_dist_real, smpl_thck, print_msg=False):
    B_coeff = const.dirac_const / (smpl_thck * d_dist_real)     # the only place where pixel size is significant
    d_phase = ph2 - ph1                                         # consider sign of magnetic field
    B_val = B_coeff * d_phase
    if print_msg:
        print('{0:.1f} nm'.format(d_dist_real * 1e9))
        print('{0:.2f} rad'.format(d_phase))
        print('B = {0:.2f} T'.format(B_val))
    return B_val

#-------------------------------------------------------------------

# calculate B from section on image
def calc_B_from_section(img, pt1, pt2, smpl_thck):
    from numpy import linalg as la
    phs = img.amPh.ph

    d_dist = la.norm(pt1 - pt2)
    d_dist_real = d_dist * img.px_dim

    n_pts_for_ls = 5
    nn_def = 4
    n_neigh_areas = 2 * (n_pts_for_ls - 1)
    n_neigh = nn_def if d_dist >= n_neigh_areas * nn_def else int(d_dist // n_neigh_areas)

    xx = np.round(np.linspace(pt1[0], pt2[0], n_pts_for_ls)).astype(np.int32)
    yy = np.round(np.linspace(pt1[1], pt2[1], n_pts_for_ls)).astype(np.int32)

    x_arr_for_ls = np.linspace(0, d_dist, n_pts_for_ls, dtype=np.float32)  # for lin. least squares calc. only the proportions between x values are important (px_sz can be skipped)
    ph_arr_for_ls = np.array([tr.calc_avg_neigh(phs, x, y, nn=n_neigh) for x, y in zip(xx, yy)])
    aa, bb = tr.lin_least_squares_alt(x_arr_for_ls, ph_arr_for_ls)

    ph1 = aa * x_arr_for_ls[0]
    ph2 = aa * x_arr_for_ls[n_pts_for_ls - 1]
    # ph1_avg = tr.calc_avg_neigh(curr_phs, pt1[0], pt1[1], nn=10)
    # ph2_avg = tr.calc_avg_neigh(curr_phs, pt2[0], pt2[1], nn=10)
    calc_B(ph1, ph2, d_dist_real, smpl_thck, print_msg=True)

#-------------------------------------------------------------------

def calc_Bxy_maps(img, smpl_thck):
    B_coeff = const.dirac_const / smpl_thck

    dx, dy = np.gradient(img.amPh.ph, img.px_dim)
    # B_sign = np.sign(dx)
    # B_field = B_sign * np.sqrt(dx * dx + dy * dy) * B_coeff
    Bx = B_coeff * dx
    By = B_coeff * dy

    Bx_img = imsup.ImageExp(img.height, img.width)
    Bx_img.amPh.ph = np.copy(Bx)
    Bx_img.name = 'Bx_from_{0}'.format(img.name)

    By_img = imsup.ImageExp(img.height, img.width)
    By_img.amPh.ph = np.copy(By)
    By_img.name = 'By_from_{0}'.format(img.name)

    return Bx_img, By_img

#-------------------------------------------------------------------

def calc_B_polar_from_orig_r(img, orig_xy, r1, smpl_thck, orig_is_pt1=False, ang0=0.0, n_r=3, addn_str=''):
    phs = img.amPh.ph
    px_sz = img.px_dim

    min_xy = [x - n_r * r1 for x in orig_xy]
    max_xy = [x + n_r * r1 for x in orig_xy]
    if min_xy[0] < 0 or min_xy[1] < 0 or max_xy[0] > img.width or max_xy[1] > img.height:
        print('One of the circular areas will go beyond edges of the image. Select new area or lower the number of radius iterations.')
        return 'f'

    r_values = [ (n + 1) * r1 for n in range(n_r) ]
    print('r = {0}'.format(r_values))

    ang1 = ang0 - np.pi / 2.0
    ang2 = ang0 + np.pi / 2.0
    n_ang = 60
    ang_arr = np.linspace(ang1, ang2, n_ang, dtype=np.float32)
    angles = [ np.copy(ang_arr) for _ in range(n_r) ]
    B_values = [ [] for _ in range(n_r) ]

    n_pts_for_ls = 5
    nn_def = 4
    n_neigh_areas = 2 * (n_pts_for_ls - 1)

    for r, r_idx in zip(r_values, range(n_r)):
        d_dist = r if orig_is_pt1 else 2 * r
        d_dist_real = d_dist * px_sz
        n_neigh = nn_def if d_dist >= n_neigh_areas * nn_def else int(d_dist // n_neigh_areas)

        x_arr_for_ls = np.linspace(0, d_dist, n_pts_for_ls, dtype=np.float32)

        for ang, a_idx in zip(angles[r_idx], range(n_ang)):
            sin_cos = np.array([np.cos(ang), -np.sin(ang)])         # -sin(ang), because y increases from top to bottom of an image
            new_pt1 = orig_xy if orig_is_pt1 else np.round(orig_xy - r * sin_cos).astype(np.int32)
            new_pt2 = np.round(orig_xy + r * sin_cos).astype(np.int32)
            x1, y1 = new_pt1
            x2, y2 = new_pt2

            xx = np.round(np.linspace(x1, x2, n_pts_for_ls)).astype(np.int32)
            yy = np.round(np.linspace(y1, y2, n_pts_for_ls)).astype(np.int32)

            # ph_arr_for_ls = np.array([ curr_phs[y, x] for y, x in zip(yy, xx) ])
            ph_arr_for_ls = np.array([ tr.calc_avg_neigh(phs, x, y, nn=n_neigh) for x, y in zip(xx, yy) ])
            aa, bb = tr.lin_least_squares_alt(x_arr_for_ls, ph_arr_for_ls)

            # ph1 = tr.calc_avg_neigh(phs, x1, y1, nn=n_neigh)
            # ph2 = tr.calc_avg_neigh(phs, x2, y2, nn=n_neigh)
            ph1 = aa * x_arr_for_ls[0]
            ph2 = aa * x_arr_for_ls[n_pts_for_ls - 1]

            B_val = calc_B(ph1, ph2, d_dist_real, smpl_thck)
            if B_val < 0: angles[r_idx][a_idx] += np.pi
            B_values[r_idx].append(np.abs(B_val))

    fig = plt.figure()
    ax = fig.add_subplot(111, projection='polar')

    B_lim1, B_lim2 = 0.0, np.max(B_values) + 0.1
    for p_idx in range(n_r):
        ax.plot(angles[p_idx], np.array(B_values[p_idx]), '.-', lw=1.0, ms=3.5,
                label='r={0:.0f}px'.format(round(r_values[p_idx])))
    ax.plot(np.array([ang0, ang0 + np.pi]), np.array([B_lim2, B_lim2]), 'k--', lw=0.8)    # mark selected direction
    ax.plot(np.array([ang1, ang2]), np.array([B_lim2, B_lim2]), 'g--', lw=0.8)            # boundary between positive and negative values of B
    ax.legend(loc='upper left', bbox_to_anchor=(1, 1), fontsize=8)
    # ax.plot(angles, np.zeros(n_ang), 'g--', lw=1)
    # for ang, r in zip(angles[:n_ang:4], B_values[0][:n_ang:4]):
    #     ax.annotate('', xytext=(0.0, r_min), xy=(ang, r), arrowprops=dict(facecolor='blue', arrowstyle='->'))
    ax.set_ylim(B_lim1, B_lim2)
    ax.grid(True)

    ax.margins(0, 0)
    fig.savefig('B_pol_{0}{1}.png'.format(img.name, addn_str), dpi=300, bbox_inches='tight', pad_inches=0)
    ax.cla()
    fig.clf()
    plt.close(fig)
    print('B_pol_{0}{1}.png exported!'.format(img.name, addn_str))

    # print [Bx, By] components of resultant (projected) vector B
    max_B = np.max(B_values[0])
    max_B_angle = angles[0][np.argmax(B_values[0])] - np.pi / 2.0
    Bx, By = max_B * np.cos(max_B_angle), -max_B * np.sin(max_B_angle)
    print('B_proj = [Bx, By] = [{0:.2f}, {1:.2f}] T'.format(Bx, By))

    return max_B_angle

#-------------------------------------------------------------------

def calc_B_polar_sectors(img, orig_xy, r1, n_rows, n_cols, smpl_thck, orig_is_pt1=False, n_r=3):
    orig_pts = []
    max_B_angles = []

    for row in range(n_rows):
        for col in range(n_cols):
            new_orig_xy = np.round([ x + j*2*r1 for x, j in zip(orig_xy, [col, row]) ]).astype(np.int32)
            rc_str = '_{0}{1}'.format(row+1, col+1)
            max_B_ang = calc_B_polar_from_orig_r(img, new_orig_xy, r1, smpl_thck, orig_is_pt1, ang0=0.0, n_r=n_r, addn_str=rc_str)
            state = max_B_ang
            if state == 'f':
                return orig_pts
            else:
                max_B_angles.append(max_B_ang)
            orig_pts.append(list(new_orig_xy))

    # export phase image with max. B vectors
    fig, ax = plt.subplots()
    im = ax.imshow(img.amPh.ph, cmap=plt.cm.get_cmap('jet'))
    cbar = fig.colorbar(im)
    cbar.set_label('phase shift [rad]')

    for [x, y], dir_ang in zip(orig_pts, max_B_angles):
        dx, dy = r1 * np.cos(dir_ang), -r1 * np.sin(dir_ang)
        ax.arrow(x, y, dx, dy, fc='k', ec='k', lw=0.6, head_width=12, head_length=20, length_includes_head=True)
        circ = Circle((x, y), r1, fill=False, ec='orange', ls='--', lw=0.6)
        ax.add_patch(circ)

    out_f = '{0}_+max_B_vec.png'.format(img.name)
    ax.axis('off')
    ax.margins(0, 0)
    ax.xaxis.set_major_locator(plt.NullLocator())
    ax.yaxis.set_major_locator(plt.NullLocator())
    fig.savefig(out_f, dpi=300, bbox_inches='tight', pad_inches=0)
    ax.cla()
    fig.clf()
    plt.close(fig)
    print('Phase image with max. B vectors exported: {0}'.format(out_f))
    # ---

    return orig_pts

#-------------------------------------------------------------------

def calc_B_polar_from_area(frag, smpl_thck):
    px_sz = frag.px_dim

    ang1, ang2 = 0.0, np.pi
    n_ang = 60
    angles = np.linspace(ang1, ang2, n_ang, dtype=np.float32)
    min_cc = tr.det_crop_coords_after_ski_rotation(frag.width, 45)  # non-zero area is the smallest for image rotated by 45 deg

    B_coeff = const.dirac_const / smpl_thck
    B_means = []

    for ang, a_idx in zip(angles, range(n_ang)):
        frag_rot = tr.rotate_image_ski(frag, imsup.Degrees(ang))
        roi_ph = np.copy(frag_rot.amPh.ph[min_cc[0]:min_cc[2], min_cc[1]:min_cc[3]])
        dx, dy = np.gradient(roi_ph, px_sz)
        B_mean = np.mean(B_coeff * dx)
        if B_mean < 0: angles[a_idx] += np.pi
        B_means.append(np.abs(B_mean))

    fig = plt.figure()
    ax = fig.add_subplot(111, projection='polar')

    B_lim1, B_lim2 = 0.0, np.max(B_means) + 0.1
    ax.plot(angles, np.array(B_means), '.-', lw=1.0, ms=3.5)
    ax.plot(np.array([ang1, ang2]), np.array([B_lim2, B_lim2]), 'g--', lw=0.8)  # boundary between positive and negative values of B
    ax.set_ylim(B_lim1, B_lim2)
    ax.grid(True)

    ax.margins(0, 0)
    fig.savefig('B_pol_{0}.png'.format(frag.name), dpi=300, bbox_inches='tight', pad_inches=0)
    ax.cla()
    fig.clf()
    plt.close(fig)
    print('B_pol_{0}.png exported!'.format(frag.name))