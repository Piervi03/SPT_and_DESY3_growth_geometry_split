from __future__ import division
import numpy as np



def convolve_HMF_2obs_fixedkernel(dN_dlnM, kernel):
    """Convolve 1d HMF into the 2d obs-obs space for a fixed kernel.
    Normalization is not conserved!"""
    kernel_shape_ = kernel.shape
    assert kernel_shape_[0]%2==0, "Kernel must be of even length!"
    assert kernel_shape_[1]%2==0, "Kernel must be of even length!"
    len_HMF = len(dN_dlnM)
    shape_out_ = np.array([len_HMF, len_HMF]) + np.array(kernel_shape_)
    res_ = np.zeros(shape_out_)
    for i in range(len_HMF):
        res_[i:i+kernel_shape_[0], i:i+kernel_shape_[1]]+= dN_dlnM[i] * kernel
    res_out_ = res_[kernel_shape_[0]//2:-kernel_shape_[0]//2, kernel_shape_[1]//2:-kernel_shape_[1]//2]
    return res_out_


def convolve_HMF_3obs_fixedkernel(dN_dlnM, kernel):
    """Convolve 1d HMF into the 3d obs-obs-obs space for a fixed kernel.
    Normalization is not conserved!"""
    kernel_shape_ = kernel.shape
    assert kernel_shape_[0]%2==0, "Kernel must be of even length!"
    assert kernel_shape_[1]%2==0, "Kernel must be of even length!"
    assert kernel_shape_[2]%2==0, "Kernel must be of even length!"
    len_HMF_ = len(dN_dlnM)
    shape_out_ = np.array([len_HMF_, len_HMF_, len_HMF_]) + np.array(kernel_shape_)
    res_ = np.zeros(shape_out_)
    for i in range(len_HMF_):
    res_[i:i+kernel_shape_[0], i:i+kernel_shape_[1], i:i+kernel_shape_[2]]+= dN_dlnM[i] * kernel
    res_out_ = res_[kernel_shape_[0]//2:-kernel_shape_[0]//2, kernel_shape_[1]//2:-kernel_shape_[1]//2, kernel_shape_[2]//2:-kernel_shape_[2]//2]
    return res_out_
