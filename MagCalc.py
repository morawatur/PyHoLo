import numpy as np

import Constants as const
import Transform as tr

import matplotlib.pyplot as plt

#-------------------------------------------------------------------

def calc_B_polar_from_2_tot_phs(img1, img2, pt1, pt2, smpl_thck, pt1_is_orig=True, n_r=3, px_sz=1.0):
    from numpy import linalg as la
    phs1 = img1.amPh.ph
    phs2 = img2.amPh.ph
    d_dist = la.norm(pt1 - pt2)

    if pt1_is_orig:
        orig_xy = pt1
        r1 = int(d_dist)
    else:
        orig_xy = np.array([int(np.mean((pt1[0], pt2[0]))), int(np.mean((pt1[1], pt2[1])))])
        r1 = int(d_dist // 2)

    min_xy = [x - n_r * r1 for x in orig_xy]
    max_xy = [x + n_r * r1 for x in orig_xy]
    if min_xy[0] < 0 or min_xy[1] < 0 or max_xy[0] > img1.width or max_xy[1] > img1.height:
        print(
            'One of the circular areas will go beyond edges of the image. Select new area or lower the number of radius iterations.')
        return

    r_values = [ (n + 1) * r1 for n in range(n_r) ]
    print('r = {0}'.format(r_values))

    ang0 = -np.arctan2(pt2[1] - pt1[1], pt2[0] - pt1[0])
    ang1 = ang0 - np.pi / 2.0
    ang2 = ang0 + np.pi / 2.0
    n_ang = 60
    ang_arr = np.linspace(ang1, ang2, n_ang, dtype=np.float32)
    angles = [ np.copy(ang_arr) for _ in range(n_r) ]

    B_coeff = const.dirac_const / (smpl_thck * d_dist * px_sz)
    d_B_values = [ [] for _ in range(n_r) ]
    x_arr_for_ls = np.linspace(0, d_dist * px_sz, 5, dtype=np.float32)

    for r, r_idx in zip(r_values, range(n_r)):
        nn_for_ls = 4 if r >= 16 else r // 4

        for ang, a_idx in zip(angles[r_idx], range(n_ang)):
            sin_cos = np.array([np.cos(ang), -np.sin(ang)])  # -sin(ang), because y increases from top to bottom of an image
            new_pt1 = pt1 if pt1_is_orig else np.array(orig_xy - r * sin_cos).astype(np.int32)
            new_pt2 = np.array(orig_xy + r * sin_cos).astype(np.int32)
            x1, y1 = new_pt1
            x2, y2 = new_pt2

            xx = np.linspace(x1, x2, 5, dtype=np.int32)
            yy = np.linspace(y1, y2, 5, dtype=np.int32)

            ph_arr1_for_ls = [tr.calc_avg_neigh(phs1, x, y, nn=nn_for_ls) for x, y in zip(xx, yy)]
            ph_arr2_for_ls = [tr.calc_avg_neigh(phs2, x, y, nn=nn_for_ls) for x, y in zip(xx, yy)]

            a1, b1 = tr.LinLeastSquares(x_arr_for_ls, ph_arr1_for_ls)
            a2, b2 = tr.LinLeastSquares(x_arr_for_ls, ph_arr2_for_ls)

            d_phase1 = a1 * (x_arr_for_ls[4] - x_arr_for_ls[0])
            d_phase2 = a2 * (x_arr_for_ls[4] - x_arr_for_ls[0])

            d_B_val = B_coeff * (d_phase2 - d_phase1)
            if d_B_val < 0: angles[r_idx][a_idx] += np.pi
            d_B_values[r_idx].append(np.abs(d_B_val))

    fig = plt.figure()
    ax = fig.add_subplot(111, projection='polar')

    B_min, B_max = 0.0, np.max(d_B_values) + 0.1
    # B_min, B_max = 0.0, const.temp_B_max_for_polar_plot
    for p_idx in range(n_r):
        ax.plot(angles[p_idx], np.array(d_B_values[p_idx]), '.-', lw=1.0, ms=3.5, label='r={0}px'.format(r_values[p_idx]))
    ax.plot(np.array([ang0, ang0 + np.pi]), np.array([B_max, B_max]), 'k--', lw=0.8)    # mark selected direction
    ax.plot(np.array([ang1, ang2]), np.array([B_max, B_max]), 'g--', lw=0.8)            # boundary between positive and negative values of B
    ax.legend(loc='upper left', bbox_to_anchor=(1, 1), fontsize=8)
    ax.set_ylim(B_min, B_max)
    ax.grid(True)

    plt.margins(0, 0)
    plt.savefig('dB_pol_{0}-{1}.png'.format(img2.name, img1.name), dpi=300, bbox_inches='tight', pad_inches=0)
    plt.close(fig)
    print('dB_pol_{0}-{1}.png exported!'.format(img2.name, img1.name))