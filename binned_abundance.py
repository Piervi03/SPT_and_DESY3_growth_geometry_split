import numpy as np
from multiprocessing import Pool
from scipy.special import ndtr

import scaling_relations


def execute(HMF, cosmology, scaling, SPT_survey_tab, z_bins, SNR_bins, NPROC=0):
    """Returns number of clusters within `z_bins` and `SNR_bins` over the whole survey."""
    lndN_dz_dlnzeta_unitSolidAng = {}
    for tmp in ['SZ_lambdacut_shallow', 'SZ_lambdacut_deep', 'SZ']:
        lndN_dz_dlnzeta_unitSolidAng[tmp] = np.log(scaling_relations.dlnM_dlnobs('zeta', scaling)) + HMF['%s_dNdlnM' % tmp]
    if NPROC == 0:
        N_field = np.array([process_field(SPT_survey_tab[i],
                                          HMF['z_arr'], HMF['lnM_arr'], lndN_dz_dlnzeta_unitSolidAng,
                                          scaling, cosmology,
                                          z_bins, SNR_bins)
                            for i in range(len(SPT_survey_tab))])
    else:
        with Pool(processes=NPROC) as pool:
            N_field = pool.starmap(process_field,
                                   [(SPT_survey_tab[i],
                                     HMF['z_arr'], HMF['lnM_arr'], lndN_dz_dlnzeta_unitSolidAng,
                                     scaling, cosmology,
                                     z_bins, SNR_bins)
                                    for i in range(len(SPT_survey_tab))])
    N_survey = np.sum(N_field, axis=0)
    return N_survey


def process_field(SPT_field,
                  z_arr, lnM_arr, lndN_dz_dlnzeta_unitSolidAng,
                  scaling, cosmology,
                  z_bins, SNR_bins):
    """Returns number of clusters within `z_bins` and `SNR_bins` for the given SPT field."""
    # dN/dz/dln(zeta)
    if SPT_field['FIELD'] == 'sptpol_500d_MCMF':
        tmp = 'SZ_lambdacut_deep'
    elif '_MCMF' in SPT_field['FIELD']:
        tmp = 'SZ_lambdacut_shallow'
    else:
        tmp = 'SZ'
    lndN_dz_dlnzeta = lndN_dz_dlnzeta_unitSolidAng[tmp] + np.log(SPT_field['AREA'] * (np.pi/180)**2)
    # zeta-mass relation (depends on survey)
    if '3G' in SPT_field['FIELD']:
        SPTsurvey = '3G'
    elif '500d' in SPT_field['FIELD']:
        SPTsurvey = '500d'
    elif '_sptpol' in SPT_field['FIELD']:
        SPTsurvey = 'ECS'
    else:
        SPTsurvey = 'SZ'
    lnzeta_m = (np.log(SPT_field['GAMMA'])
                + scaling_relations.lnmass2lnobs('zeta', lnM_arr[None, :], z_arr[:, None],
                                                 scaling, cosmology, SPTsurvey=SPTsurvey))
    if '_sptpol' in SPT_field['FIELD']:
        lnzeta_m += np.log(scaling['SPECS_calib'])
    # zeta_min
    lndN_dz_dlnzeta[lnzeta_m < np.log(scaling['zeta_min'])] = -np.inf
    # xi-zeta relation
    xi = scaling_relations.zeta2xi(np.exp(lnzeta_m))
    # dN/dxi = dN/dln(zeta) * dln(zeta)/dxi
    lndN_dz_dxi = lndN_dz_dlnzeta + np.log(scaling_relations.dlnzeta_dxi_given_xi(xi))
    # Integrate
    xi_bins = scaling_relations.zeta2xi(SNR_bins/SPT_field['GAMMA'])
    num_z_bins = len(z_bins) - 1
    num_SNR_bins = len(SNR_bins) - 1
    N = np.empty(num_z_bins * num_SNR_bins)
    for i in range(num_z_bins):
        z_idx = (z_arr >= z_bins[i]) & (z_arr < z_bins[i+1])
        for j in range(num_SNR_bins):
            P_xi = ndtr(xi[z_idx, :] - xi_bins[j+1]) - ndtr(xi[z_idx, :] - xi_bins[j])
            lnitg = lndN_dz_dxi[z_idx, :] * np.log(P_xi)
            dN_dz = np.sum(np.exp(.5 * (lnitg[:, :-1] + lnitg[:, 1:])) * (xi[z_idx, 1:] - xi[z_idx, :-1]), axis=1)
            N[i*num_SNR_bins + j] = np.sum(.5 * (dN_dz[:-1] + dN_dz[1:]) * (z_arr[z_idx][1:] - z_arr[z_idx][:-1]))
    return N
