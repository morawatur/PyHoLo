import numpy as np

import Constants as const
import Transform as tr

import matplotlib.pyplot as plt
from matplotlib.patches import Circle

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
        n_neigh = nn_def if d_dist >= n_neigh_areas * nn_def else int(d_dist // n_neigh_areas)

        B_coeff = const.dirac_const / (smpl_thck * d_dist * px_sz)
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
            aa, bb = tr.LinLeastSquaresAlt(x_arr_for_ls, ph_arr_for_ls)

            # d_phase = tr.calc_avg_neigh(phs, x2, y2, nn=n_neigh) - tr.calc_avg_neigh(phs, x1, y1, nn=n_neigh)
            d_phase = aa * (x_arr_for_ls[n_pts_for_ls - 1] - x_arr_for_ls[0])
            B_val = B_coeff * d_phase
            if B_val < 0: angles[r_idx][a_idx] += np.pi
            B_values[r_idx].append(np.abs(B_val))

    fig = plt.figure()
    ax = fig.add_subplot(111, projection='polar')

    B_min, B_max = 0.0, np.max(B_values) + 0.1
    # B_min, B_max = 0.0, const.temp_B_max_for_polar_plot
    for p_idx in range(n_r):
        ax.plot(angles[p_idx], np.array(B_values[p_idx]), '.-', lw=1.0, ms=3.5,
                label='r={0:.0f}px'.format(round(r_values[p_idx])))
    ax.plot(np.array([ang0, ang0 + np.pi]), np.array([B_max, B_max]), 'k--', lw=0.8)    # mark selected direction
    ax.plot(np.array([ang1, ang2]), np.array([B_max, B_max]), 'g--', lw=0.8)            # boundary between positive and negative values of B
    ax.legend(loc='upper left', bbox_to_anchor=(1, 1), fontsize=8)
    # ax.plot(angles, np.zeros(n_ang), 'g--', lw=1)
    # for ang, r in zip(angles[:n_ang:4], B_values[:n_ang:4]):
    #     ax.annotate('', xytext=(0.0, r_min), xy=(ang, r), arrowprops=dict(facecolor='blue', arrowstyle='->'))
    ax.set_ylim(B_min, B_max)
    ax.grid(True)

    ax.margins(0, 0)
    fig.savefig('B_pol_{0}{1}.png'.format(img.name, addn_str), dpi=300, bbox_inches='tight', pad_inches=0)
    ax.cla()
    fig.clf()
    plt.close(fig)
    print('B_pol_{0}{1}.png exported!'.format(img.name, addn_str))

    max_B_angle = angles[0][np.argmax(B_values[0])] - np.pi / 2.0
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
        ax.arrow(x, y, dx, dy, fc='k', ec='k', lw=0.6, head_width=3, head_length=6, length_includes_head=True)
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