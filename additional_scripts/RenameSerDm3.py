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

# ----------------------------------------------------------------------

import os
import glob

path = 'C:\\img_dir\\'
dm3_files = sorted(glob.glob(path + '*.dm3'))
num_of_files = len(dm3_files)
description_file = open(path + 'info.txt', 'w')

new_ser_name = 'ser'

for file, idx in zip(dm3_files, range(num_of_files)):
    # old_fname = file.replace(path, '').replace('.dm3', '')
    old_fname = file.replace(path, '')
    new_fname = '{0}_{1}.dm3'.format(new_ser_name, idx + 1)
    if idx < 9:
        new_fname = new_fname[:-5] + '0' + new_fname[-5:]
    description_file.write('{0}\t{1}\tamp\n'.format(new_fname.replace('.dm3', ''), old_fname.replace('.dm3', '')))
    os.rename(file, path + new_fname)

description_file.close()