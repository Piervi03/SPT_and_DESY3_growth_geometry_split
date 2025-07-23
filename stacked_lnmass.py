import numpy as np

import P_Mwl_given_SZrichness

def execute(HMF,
            cosmology, scaling,
            SPT_survey_tab,
            survey_cut_lambda, richness_scatter_model,
            z_bins, SNR_bins,):
    # Draws of all observables from the halo mass function
    z, xi, SNR, lnrichness, lnMwl, lnw = P_Mwl_given_SZrichness.execute(HMF,
                                                                        cosmology, scaling,
                                                                        SPT_survey_tab,
                                                                        survey_cut_lambda, richness_scatter_model,
                                                                        NPROC=0))
    # Normalize the weights
    lnw -= np.amax(lnw)
    # Only bins with upper limits above xi cut
    num_z_bins = len(z_bins) - 1
    num_SNR_bins = len(SNR_bins) - 1
    lnMwl_mean = np.zeros(num_z_bins * num_SNR_bins)
    for i in range(num_z_bins):
        for j in SNR_bin_idx:
            idx = (z >= z_bins[i]) & (z < z_bins[i+1]) & (SNR >= SNR_bins[j]) & (SNR < SNR_bins[j+1])
            lnMwl_mean[i, j] = np.sum(lnMwl[idx] * np.exp(lnw[idx])) / np.sum(np.exp(lnw[idx]))
    return lnMwl_mean
