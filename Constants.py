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

img_dim = 2048
disp_dim = 768
min_px_threshold = 0.2
max_px_threshold = 1.8
n_div_for_warp = 8
disp_name_max_len = 60

input_dir = 'input'
output_dir = 'output'

aperture = 64                       # aperture diameter
hann_win = 72                       # side length of Hann. window
smooth_width = 5                    # width of aperture edge smoothing
min_COM_roi_hlf_edge = 5

ew_lambda = 1.968749e-12            # m (300kV)
planck_const = 6.626070e-34         # J*s
dirac_const = 6.582118e-16          # eV*s
light_speed = 2.997925e8            # m/s

el_rest_mass = 9.109382e-31         # kg
el_charge = 1.602177e-19            # C

corr_arr_max_shift = 20

#-------------------------------------------------------------------

def check_output_dir():
    import os
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)