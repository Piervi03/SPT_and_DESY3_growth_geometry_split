import numpy as np
from multiprocessing import Pool
from scipy.special import ndtr as ndtr

import scaling_relations


def execute(HMF,
            cosmology, scaling,
            SPT_survey_tab,
            z_bins, SNR_bins,
            NPROC=0):
    """Returns number of clusters within `z_bins` and `SNR_bins` over the whole survey."""
    if NPROC == 0:
        N_field = np.array([process_field(SPT_survey_tab[i],
                                          HMF,
                                          scaling, cosmology,
                                          z_bins, SNR_bins)
                            for i in range(len(SPT_survey_tab))])
    else:
        with Pool(processes=NPROC) as pool:
            N_field = pool.starmap(process_field,
                                   [(SPT_survey_tab[i],
                                     HMF,
                                     scaling, cosmology,
                                     z_bins, SNR_bins)
                                    for i in range(len(SPT_survey_tab))])
    N_survey = np.sum(N_field, axis=0)
    return N_survey


def process_field(SPTfield,
                  HMF,
                  scaling, cosmology,
                  z_bins, SNR_bins):
    """Returns number of clusters within `z_bins` and `SNR_bins` for the given SPT field."""
    # dN/dz/dln(zeta)
    if SPTfield['LAMBDA_MIN'] not in ['none', 'None', 'NONE']:
        tmp = 'SZ_lambdacut_' + SPTfield['LAMBDA_MIN']
    else:
        tmp = 'SZ'
    lndN_dz_dlnzeta = (HMF['{}_lndNdlnM'.format(tmp)]
                       + np.log(scaling_relations.dlnM_dlnobs('zeta', scaling)
                                * SPTfield['AREA'] * (np.pi/180)**2))
    # zeta-mass relation (depends on field)
    lnzeta_m = scaling_relations.lnmass2lnobs('zeta', HMF['lnM_arr'][None, :], HMF['z_arr'][:, None],
                                              scaling, cosmology, SPTfield=SPTfield)
    # zeta_min
    lndN_dz_dlnzeta[lnzeta_m < np.log(scaling['zeta_min'])] = -np.inf
    # xi-zeta relation
    xi = scaling_relations.zeta2xi(np.exp(lnzeta_m))
    # dN/dxi = dN/dln(zeta) * dln(zeta)/dxi
    lndN_dz_dxi = lndN_dz_dlnzeta + np.log(scaling_relations.dlnzeta_dxi_given_xi(xi))
    # N within z and SNR bins
    xi_bins = scaling_relations.zeta2xi(SNR_bins * SPTfield['GAMMA'])
    # Only bins with upper limits above xi cut
    SNR_bin_idx = (SPTfield['XI_MIN'] < xi_bins).nonzero()[0] - 1
    num_z_bins = len(z_bins) - 1
    num_SNR_bins = len(SNR_bins) - 1
    N = np.zeros(num_z_bins * num_SNR_bins)
    for i in range(num_z_bins):
        z_idx = (HMF['z_arr'] >= z_bins[i]) & (HMF['z_arr'] < z_bins[i+1])
        for j in SNR_bin_idx:
            # xi cut is within SNR bin
            xi_lo = np.amax([SPTfield['XI_MIN'], xi_bins[j]])
            # contribution of dN/dxi to bin
            P_xi = ndtr(xi_bins[j+1] - xi[z_idx, :]) - ndtr(xi_lo - xi[z_idx, :])
            # Integrate over xi and z
            with np.errstate(divide='ignore'):
                lnitg = lndN_dz_dxi[z_idx, :] + np.log(P_xi)
            dN_dz = np.sum(np.exp(.5 * (lnitg[:, :-1] + lnitg[:, 1:])) * (xi[z_idx, 1:] - xi[z_idx, :-1]), axis=1)
            N[i*num_SNR_bins + j] = np.sum(.5 * (dN_dz[:-1] + dN_dz[1:]) * (HMF['z_arr'][z_idx][1:] - HMF['z_arr'][z_idx][:-1]))
    return N
