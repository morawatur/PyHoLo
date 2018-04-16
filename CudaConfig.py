import numpy as np

# -------------------------------------------------------------------

def DetermineCudaConfig(dim):
    blockDim = 32, 32
    gridDim = np.ceil(dim / blockDim[0]).astype(np.int32), np.ceil(dim / blockDim[1]).astype(np.int32)
    return blockDim, gridDim

# -------------------------------------------------------------------

def DetermineCudaConfigNew(dims):
    blockDim = 32, 32
    gridDim = np.ceil(dims[0] / blockDim[0]).astype(np.int32), np.ceil(dims[1] / blockDim[1]).astype(np.int32)
    return blockDim, gridDim