import numpy as np


def convolve_HMF_1obs_varkernel(dN_dlnM, Delta_lnM, kernels, Nbins):
    """Convolve HMF with varying kernels."""
    len_HMF = len(dN_dlnM)
    kernel_shapes = np.array([kernels[i].shape for i in range(len(kernels))])
    if __debug__:
        assert len(kernels)==len_HMF, "Need as many kernels as entries in HMF"
        assert np.all(kernel_shapes%2==1), "Kernels must be of uneven length"
    # Compute padding
    pad_lo = np.amax(Nbins[:,0]-range(len_HMF))
    pad_hi = np.amax(Nbins[:,1]-range(len_HMF)[::-1])
    # Sum up contributions
    res = np.zeros(len_HMF+pad_lo+pad_hi)
    for i in range(len_HMF):
        idx_lo = i+pad_lo-Nbins[i,0]
        idx_hi = i+pad_lo+Nbins[i,1]+1
        res[idx_lo:idx_hi]+= dN_dlnM[i] * kernels[i]
    # Remove padding
    res_out = Delta_lnM * res[pad_lo:-pad_hi]
    return res_out


def convolve_HMF_2obs_fixedkernel(dN_dlnM, Delta_lnM, kernel, Nbins_0, Nbins_1):
    """Convolve 1d HMF into the 2d obs-obs space for a fixed kernel."""
    kernel_shape = np.array(kernel.shape)
    if __debug__:
        assert np.all(kernel_shape%2==1), "Kernel must be of uneven length"
        assert np.all(kernel_shape-[Nbins_0[0]+Nbins_0[1], Nbins_1[0]+Nbins_1[1]]==1), "Kernel and Nbins do not match"
    len_HMF = len(dN_dlnM)
    shape_out = np.array([len_HMF, len_HMF]) + [Nbins_0[0]+Nbins_0[1], Nbins_1[0]+Nbins_1[1]]
    res = np.zeros(shape_out)
    for i in range(len_HMF):
        res[i:i+kernel_shape[0], i:i+kernel_shape[1]]+= dN_dlnM[i] * kernel
    res_out = Delta_lnM * res[Nbins_0[0]:-Nbins_0[1], Nbins_1[0]:-Nbins_1[1]]
    return res_out


def convolve_HMF_3obs_fixedkernel(dN_dlnM, Delta_lnM, kernel, Nbins_0, Nbins_1, Nbins_2):
    """Convolve 1d HMF into the 3d obs-obs-obs space for a fixed kernel."""
    kernel_shape = kernel.shape
    if __debug__:
        assert np.all(kernel_shape%2==1), "Kernel must be of uneven length"
        assert np.all(kernel_shape-[Nbins_0[0]+Nbins_0[1], Nbins_1[0]+Nbins_1[1], Nbins_2[0]+Nbins_2[1]]==1), "Kernel and Nbins do not match"
    len_HMF = len(dN_dlnM)
    shape_out = np.array([len_HMF, len_HMF, len_HMF]) + [Nbins_0[0]+Nbins_0[1], Nbins_1[0]+Nbins_1[1], Nbins_2[0]+Nbins_2[1]]
    res = np.zeros(shape_out)
    for i in range(len_HMF):
        res[i:i+kernel_shape[0], i:i+kernel_shape[1], i:i+kernel_shape[2]]+= dN_dlnM[i] * kernel
    res_out = Delta_lnM * res[Nbins_0[0]:-Nbins_0[1], Nbins_1[0]:-Nbins_1[1], Nbins_2[0]:-Nbins_2[1]]
    return res_out


def convolve_HMF_2obs_varkernel(dN_dlnM, Delta_lnM, kernels, Nbins_0, Nbins_1):
    """Convolve 1d HMF into the 2d obs-obs space for varying kernels."""
    len_HMF = len(dN_dlnM)
    kernel_shapes = np.array([kernels[i].shape for i in range(len(kernels))])
    if __debug__:
        assert len(kernels)==len_HMF, "Need as many kernels as entries in HMF"
        assert np.all(kernel_shapes%2==1), "Kernels must be of uneven length"
    # Compute padding
    pad_0_lo = np.amax(Nbins_0[:,0]-range(len_HMF))
    pad_0_hi = np.amax(Nbins_0[:,1]-range(len_HMF)[::-1])
    pad_1_lo = np.amax(Nbins_1[:,0]-range(len_HMF))
    pad_1_hi = np.amax(Nbins_1[:,1]-range(len_HMF)[::-1])
    # Sum up contributions
    shape_out = [len_HMF+pad_0_lo+pad_0_hi, len_HMF+pad_1_lo+pad_1_hi]
    res = np.zeros(shape_out)
    for i in range(len_HMF):
        idx_0_lo = i+pad_0_lo-Nbins_0[i,0]
        idx_0_hi = i+pad_0_lo+Nbins_0[i,1]+1
        idx_1_lo = i+pad_1_lo-Nbins_1[i,0]
        idx_1_hi = i+pad_1_lo+Nbins_1[i,1]+1
        res[idx_0_lo:idx_0_hi, idx_1_lo:idx_1_hi]+= dN_dlnM[i] * kernels[i]
    # Remove padding
    res_out = Delta_lnM * res[pad_0_lo:-pad_0_hi, pad_1_lo:-pad_1_hi]
    return res_out


def convolve_HMF_3obs_varkernel(lndN_dlnM, Delta_lnM, lnkernels, Nbins_0, Nbins_1, Nbins_2):
    """Convolve 1d HMF into the 3d obs space for varying kernels."""
    len_HMF = len(lndN_dlnM)
    kernel_shapes = np.array([lnkernels[i].shape for i in range(len(lnkernels))])
    if __debug__:
        assert len(lnkernels)==len_HMF, "Need as many kernels as entries in HMF"
        assert np.all(kernel_shapes%2==1), "Kernels must be of uneven length"
    # Compute padding
    pad_0_lo = np.amax(Nbins_0[:,0]-range(len_HMF))
    pad_0_hi = np.amax(Nbins_0[:,1]-range(len_HMF)[::-1])
    pad_1_lo = np.amax(Nbins_1[:,0]-range(len_HMF))
    pad_1_hi = np.amax(Nbins_1[:,1]-range(len_HMF)[::-1])
    pad_2_lo = np.amax(Nbins_2[:,0]-range(len_HMF))
    pad_2_hi = np.amax(Nbins_2[:,1]-range(len_HMF)[::-1])
    shape_out = [len_HMF+pad_0_lo+pad_0_hi, len_HMF+pad_1_lo+pad_1_hi, len_HMF+pad_2_lo+pad_2_hi]
    # Sum up contributions
    res = np.zeros(shape_out)
    for i in range(len_HMF):
        idx_0_lo = i+pad_0_lo-Nbins_0[i,0]
        idx_0_hi = i+pad_0_lo+Nbins_0[i,1]+1
        idx_1_lo = i+pad_1_lo-Nbins_1[i,0]
        idx_1_hi = i+pad_1_lo+Nbins_1[i,1]+1
        idx_2_lo = i+pad_2_lo-Nbins_2[i,0]
        idx_2_hi = i+pad_2_lo+Nbins_2[i,1]+1
        res[idx_0_lo:idx_0_hi, idx_1_lo:idx_1_hi, idx_2_lo:idx_2_hi]+= np.exp(lndN_dlnM[i] + lnkernels[i])
    # Remove padding
    res_out = Delta_lnM * res[pad_0_lo:-pad_0_hi, pad_1_lo:-pad_1_hi, pad_2_lo:-pad_2_hi]
    return res_out
