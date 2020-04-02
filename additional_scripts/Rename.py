import os
import glob

path = 'C:\\img_dir\\'
dm3_files = glob.glob(path + '*.dm3')
num_of_files = len(dm3_files)
description_file = open(path + 'info.txt', 'w')

new_ser_name = 'ser'

for file, idx in zip(dm3_files, range(num_of_files)):
    # old_fname = file.replace(path, '').replace('.dm3', '')
    old_fname = file.replace(path, '')
    new_fname = '{0}_{1}.dm3'.format(new_ser_name, idx + 1)
    if idx < 9:
        new_fname = new_fname[:-5] + '0' + new_fname[-5:]
    description_file.write('{0}\t{1}\tamp\n'.format(new_fname, old_fname))
    os.rename(file, path + new_fname)

description_file.close()