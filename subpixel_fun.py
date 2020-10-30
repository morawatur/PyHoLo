import numpy as np

def subpixel_shift(arr, shx, shy):
    sz = arr.shape[0]
    wx = np.abs(shx)
    wy = np.abs(shy)

    arr_shx_1px = np.zeros((sz, sz), dtype=np.float32)
    arr_shy_1px = np.zeros((sz, sz), dtype=np.float32)

    # x dir
    if shx > 0:
        arr_shx_1px[:, 1:] = arr[:, :sz - 1]
    else:
        arr_shx_1px[:, :sz - 1] = arr[:, 1:]

    arr_shx = wx * arr_shx_1px + (1.0-wx) * arr

    # y dir
    if shy > 0:
        arr_shy_1px[1:, :] = arr_shx[:sz - 1, :]
    else:
        arr_shy_1px[:sz - 1, :] = arr_shx[1:, :]

    arr_shy = wy * arr_shy_1px + (1.0-wy) * arr_shx

    return arr_shy

# arr1 = np.array((3, 3), dtype=np.float32)
arr1 = np.array([[9, 10, 2], [11, 5, 7], [12, 8, 6]], dtype=np.float32)
arr2 = subpixel_shift(arr1, 0.3, 0)
print(arr2)