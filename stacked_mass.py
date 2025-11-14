import numpy as np

import P_Mwl_given_SZrichness


def execute(HMF,
            cosmology, scaling,
            SPT_survey_tab,
            survey_cut_lambda, richness_scatter_model,
            z_bins, rot_mat, rot_bins_x, rot_bins_y,
            N_draws=1000000,
            N_bootstrap=0):
    # Draws of all observables from the halo mass function
    z, xi, SNR, richness, lnMwl, lnw = P_Mwl_given_SZrichness.execute(HMF,
                                                                      cosmology, scaling,
                                                                      SPT_survey_tab,
                                                                      survey_cut_lambda, richness_scatter_model,
                                                                      [z_bins[0], z_bins[-1]],
                                                                      N_draws=N_draws,
                                                                      NPROC=0)
    # Normalize the weights
    lnw -= np.amax(lnw)
    # Rotate to approximate mass-like space
    rot_obs = np.matmul(rot_mat, [np.log(SNR), np.log(richness)]).T
    # Mean in bins
    Mwl_mean = np.zeros((len(z_bins)-1, len(rot_bins_x)-1, len(rot_bins_y)-1))
    if N_bootstrap > 0:
        Mwl_err = np.zeros_like(Mwl_mean)
    for i in range(len(z_bins)-1):
        i_idx = ((z >= z_bins[i]) & (z < z_bins[i+1])).nonzero()[0]
        for j in range(len(rot_bins_x)-1):
            ij_idx = i_idx[(rot_obs[i_idx, 0] >= rot_bins_x[j]) & (rot_obs[i_idx, 0] < rot_bins_x[j+1])]
            for k in range(len(rot_bins_y)-1):
                ijk_idx = ij_idx[(rot_obs[ij_idx, 1] >= rot_bins_y[k]) & (rot_obs[ij_idx, 1] < rot_bins_y[k+1])]
                if not np.any(ijk_idx):
                    continue
                Mwl_mean[i, j, k] = np.sum(np.exp(lnMwl[ijk_idx]) * np.exp(lnw[ijk_idx])) / np.sum(np.exp(lnw[ijk_idx]))
                if N_bootstrap > 0:
                    Mwl_boot = np.zeros(N_bootstrap)
                    for b in range(N_bootstrap):
                        boot_idx = np.random.choice(ijk_idx, size=len(ijk_idx), replace=True)
                        Mwl_boot[b] = np.sum(np.exp(lnMwl[boot_idx]) * np.exp(lnw[boot_idx])) / np.sum(np.exp(lnw[boot_idx]))
                    Mwl_err[i, j, k] = np.std(Mwl_boot)
    if N_bootstrap > 0:
        return Mwl_mean, Mwl_err
    else:
        return Mwl_mean
