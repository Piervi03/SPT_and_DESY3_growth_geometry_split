from __future__ import division
import numpy as np



def convolve_HMF_2obs_fixedkernel(dN_dlnM, kernel):
    """Convolve 1d HMF into the 2d obs-obs space for a fixed kernel.
    Normalization is not conserved!"""
    kernel_shape = kernel.shape
    assert kernel_shape[0]%2==0, "Kernel must be of even length!"
    assert kernel_shape[1]%2==0, "Kernel must be of even length!"
    len_HMF = len(dN_dlnM)
    shape_out = np.array([len_HMF, len_HMF]) + np.array(kernel_shape)
    res = np.zeros(shape_out)
    for i in range(len_HMF):
        res[i:i+kernel_shape[0], i:i+kernel_shape[1]]+= dN_dlnM[i] * kernel
    res_out = res[kernel_shape[0]//2:-kernel_shape[0]//2, kernel_shape[1]//2:-kernel_shape[1]//2]
    return res_out


def convolve_HMF_3obs_fixedkernel(dN_dlnM, kernel):
    """Convolve 1d HMF into the 3d obs-obs-obs space for a fixed kernel.
    Normalization is not conserved!"""
    kernel_shape = kernel.shape
    assert kernel_shape[0]%2==0, "Kernel must be of even length!"
    assert kernel_shape[1]%2==0, "Kernel must be of even length!"
    assert kernel_shape[2]%2==0, "Kernel must be of even length!"
    _len_HMF = len(dN_dlnM)
    shape_out = np.array([_len_HMF, _len_HMF, _len_HMF]) + np.array(kernel_shape)
    res = np.zeros(shape_out)
    for i in range(_len_HMF):
        res[i:i+kernel_shape[0], i:i+kernel_shape[1], i:i+kernel_shape[2]]+= dN_dlnM[i] * kernel
    res_out = res[kernel_shape[0]//2:-kernel_shape[0]//2, kernel_shape[1]//2:-kernel_shape[1]//2, kernel_shape[2]//2:-kernel_shape[2]//2]
    return res_out


def convolve_HMF_2obs_varkernel(dN_dlnM, kernels):
    """Convolve 1d HMF into the 2d obs-obs space for varying kernels.
    Normalization is not conserved!"""
    # Validate input
    len_HMF = len(dN_dlnM)
    assert len(kernels)==len_HMF, "Need as many kernels as entries in HMF"
    kernel_shapes = np.array([kernels[i].shape for i in range(len_HMF)])
    assert np.all(kernel_shapes%2==0), "Kernels must be of even length"
    # Compute padding
    buffer_0_lo = np.amax(kernel_shapes[:,0]//2-range(len_HMF))
    buffer_0_hi = np.amax(kernel_shapes[:,0]//2-range(len_HMF)[::-1])
    buffer_1_lo = np.amax(kernel_shapes[:,1]//2-range(len_HMF))
    buffer_1_hi = np.amax(kernel_shapes[:,1]//2-range(len_HMF)[::-1])
    shape_out = [len_HMF+buffer_0_lo+buffer_0_hi, len_HMF+buffer_1_lo+buffer_1_hi]
    # Do the convolution
    res = np.zeros(shape_out)
    for i in range(len_HMF):
        idx_0_lo = i+buffer_0_lo-kernel_shapes[i][0]//2
        idx_0_hi = i+buffer_0_lo+kernel_shapes[i][0]//2
        idx_1_lo = i+buffer_1_lo-kernel_shapes[i][1]//2
        idx_1_hi = i+buffer_1_lo+kernel_shapes[i][1]//2
        res[idx_0_lo:idx_0_hi, idx_1_lo:idx_1_hi]+= dN_dlnM[i] * kernels[i]
    # Remove padding
    res_out = res[buffer_0_lo:-buffer_0_hi, buffer_1_lo:-buffer_1_hi]
    return res_out


def convolve_HMF_3obs_varkernel(dN_dlnM, kernels):
    """Convolve 1d HMF into the 3d obs space for varying kernels.
    Normalization is not conserved!"""
    # Validate input
    len_HMF = len(dN_dlnM)
    assert len(kernels)==len_HMF, "Need as many kernels as entries in HMF"
    kernel_shapes = np.array([kernels[i].shape for i in range(len_HMF)])
    assert np.all(kernel_shapes%2==0), "Kernels must be of even length"
    # Compute padding
    buffer_0_lo = np.amax(kernel_shapes[:,0]//2-range(len_HMF))
    buffer_0_hi = np.amax(kernel_shapes[:,0]//2-range(len_HMF)[::-1])
    buffer_1_lo = np.amax(kernel_shapes[:,1]//2-range(len_HMF))
    buffer_1_hi = np.amax(kernel_shapes[:,1]//2-range(len_HMF)[::-1])
    buffer_2_lo = np.amax(kernel_shapes[:,2]//2-range(len_HMF))
    buffer_2_hi = np.amax(kernel_shapes[:,2]//2-range(len_HMF)[::-1])
    shape_out = [len_HMF+buffer_0_lo+buffer_0_hi, len_HMF+buffer_1_lo+buffer_1_hi, len_HMF+buffer_2_lo+buffer_2_hi]
    # Do the convolution
    res = np.zeros(shape_out)
    for i in range(len_HMF):
        idx_0_lo = i+buffer_0_lo-kernel_shapes[i][0]//2
        idx_0_hi = i+buffer_0_lo+kernel_shapes[i][0]//2
        idx_1_lo = i+buffer_1_lo-kernel_shapes[i][1]//2
        idx_1_hi = i+buffer_1_lo+kernel_shapes[i][1]//2
        idx_2_lo = i+buffer_2_lo-kernel_shapes[i][2]//2
        idx_2_hi = i+buffer_2_lo+kernel_shapes[i][2]//2
        res[idx_0_lo:idx_0_hi, idx_1_lo:idx_1_hi, idx_2_lo:idx_2_hi]+= dN_dlnM[i] * kernels[i]
    # Remove padding
    res_out = res[buffer_0_lo:-buffer_0_hi, buffer_1_lo:-buffer_1_hi, buffer_2_lo:-buffer_2_hi]
    return res_out
