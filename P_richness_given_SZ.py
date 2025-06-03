import numpy as np
from multiprocessing import Pool
from scipy.interpolate import interp1d

import scaling_relations


def execute(catalog, HMF, cosmology, scaling, SPT_survey_tab, surveyCutRichness, NPROC=0):
    """Returns ln-likelihood of all `catalog['richness']` measurements given
     `catalog['XI']`."""
    # Richness-mass relation, valid for all SPT fields
    lnrichness_m = scaling_relations.lnmass2lnobs('richness',
                                                  HMF['lnM_arr'][None, :], HMF['z_arr'][:, None],
                                                  scaling, cosmology)
    # Cycle through SPT fields (because that affects the zeta-mass relation)
    if NPROC == 0:
        lnlike_field = np.array([process_field(SPT_survey_tab[i],
                                               catalog['XI', 'richness', 'REDSHIFT', 'FIELD'],
                                               surveyCutRichness,
                                               HMF['z_arr'], HMF['lnM_arr'], HMF['richness_SZ_lndNdlnM'],
                                               lnrichness_m,
                                               scaling, cosmology)
                                 for i in range(len(SPT_survey_tab))])
    else:
        with Pool(processes=NPROC) as pool:
            lnlike_field = pool.starmap(process_field,
                                        [(SPT_survey_tab[i],
                                          catalog['XI', 'richness', 'REDSHIFT', 'FIELD'],
                                          surveyCutRichness,
                                          HMF['z_arr'], HMF['lnM_arr'], HMF['richness_SZ_lndNdlnM'],
                                          lnrichness_m,
                                          scaling, cosmology)
                                         for i in range(len(SPT_survey_tab))])
    lnlike = np.sum(lnlike_field)
    return lnlike


def process_field(SPT_field, catalog, surveyCutRichness,
                  z_arr, lnM_arr, lndN_dz_dlnrichness_dlnzeta,
                  lnrichness_m,
                  scaling, cosmology):
    """Returns ln-likelihood of `catalog['richness']` given `catalog['XI']`
    for all clusters in `SPT_field`."""
    lnrichness_m_interp = interp1d(z_arr, lnrichness_m, axis=0)
    if SPT_field['FIELD'] == 'sptpol_500d_MCMF':
        lambda_min = surveyCutRichness['deep']
    else:
        lambda_min = surveyCutRichness['shallow']
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
    lnzeta_m_interp = interp1d(z_arr, lnzeta_m, axis=0)
    # zeta_min
    lndN_dz_dlnrichness_dlnzeta[lnzeta_m < np.log(scaling['zeta_min'])] = -np.inf
    # Account for dlnM/dlnobs
    lndN_dz_dlnrichness_dlnzeta += np.log(scaling_relations.dlnM_dlnobs('richness', scaling, z=z_arr)
                                          * scaling_relations.dlnM_dlnobs('zeta', scaling))[:, None, None]
    # xi = scaling_relations.zeta2xi(np.exp(lnzeta_m))
    lndN_dz_dlnrichness_dlnzeta_interp = interp1d(z_arr, lndN_dz_dlnrichness_dlnzeta, axis=0)
    field_idx = (catalog['FIELD'] == SPT_field['FIELD']).nonzero()[0]
    lnlike = 0.
    for clusterID in field_idx:
        # Look up dN/dlnlambda/dlnzeta, richness, lnzeta, xi
        lndN_dz_dlnrichness_dlnzeta = lndN_dz_dlnrichness_dlnzeta_interp(catalog['REDSHIFT'][clusterID])
        richness = np.exp(lnrichness_m_interp(catalog['REDSHIFT'][clusterID]))
        lnzeta = lnzeta_m_interp(catalog['REDSHIFT'][clusterID])
        xi = scaling_relations.zeta2xi(np.exp(lnzeta))
        # Integrate over -4, +3 sigma in xi
        idx = (xi > catalog['XI'][clusterID] - 4.) & (xi < catalog['XI'][clusterID] + 3.)
        lnP_xi = -.5 * (catalog['XI'][clusterID] - xi[idx])**2.
        lnitg = lnP_xi + lndN_dz_dlnrichness_dlnzeta[:, idx]
        P_lambda = np.sum(np.exp(.5 * (lnitg[:, :-1] + lnitg[:, 1:]))
                          * (lnzeta[idx][1:] - lnzeta[idx][:-1]), axis=1) / richness
        P_lambda[richness < lambda_min(catalog['REDSHIFT'][clusterID])] = 0.
        # Normalize
        P_lambda /= np.trapz(P_lambda, richness)
        # Evaluate
        lnlike += np.interp(catalog['richness'][clusterID], richness, np.log(P_lambda))
    return lnlike
