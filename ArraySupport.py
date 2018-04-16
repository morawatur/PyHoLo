import numpy as np
import CudaConfig as ccfg
from numba import cuda
import math

# -------------------------------------------------------------------

def AddTwoArrays(arr1, arr2):
    blockDim, gridDim = ccfg.DetermineCudaConfig(arr1.shape[0])
    # arrSum = cuda.device_array(arr1.shape, dtype=arr1.dtype)
    arrSum = cuda.to_device(np.zeros(arr1.shape, dtype=arr1.dtype))
    AddTwoArrays_dev[gridDim, blockDim](arr1, arr2, arrSum)
    return arrSum

# -------------------------------------------------------------------

# @cuda.jit('void(float32[:, :], float32[:, :], float32[:, :])')
@cuda.jit()
def AddTwoArrays_dev(arr1, arr2, arrSum):
    x, y = cuda.grid(2)
    if x >= arr1.shape[0] or y >= arr1.shape[1]:
        return
    arrSum[x, y] = arr1[x, y] + arr2[x, y]

# -------------------------------------------------------------------

def MultArrayByScalar(arr, scalar):
    blockDim, gridDim = ccfg.DetermineCudaConfig(arr.shape[0])
    arrRes = cuda.device_array(arr.shape, dtype=arr.dtype)
    MultArrayByScalar_dev[gridDim, blockDim](arr, scalar, arrRes)
    return arrRes

# -------------------------------------------------------------------

@cuda.jit()
def MultArrayByScalar_dev(arr, scalar, arrRes):
    x, y = cuda.grid(2)
    if x >= arr.shape[0] or y >= arr.shape[1]:
        return
    arrRes[x, y] = arr[x, y] * scalar

# -------------------------------------------------------------------

def CalcSqrtOfArray(arr):
    blockDim, gridDim = ccfg.DetermineCudaConfig(arr.shape[0])
    arrSqrt = cuda.device_array(arr.shape, dtype=arr.dtype)
    CalcSqrtOfArray_dev[gridDim, blockDim](arr, arrSqrt)
    return arrSqrt

# -------------------------------------------------------------------

@cuda.jit('void(float32[:, :], float32[:, :])')
def CalcSqrtOfArray_dev(arr, arrSqrt):
    x, y = cuda.grid(2)
    if x >= arr.shape[0] or y >= arr.shape[1]:
        return
    arrSqrt[x, y] = math.sqrt(arr[x, y])