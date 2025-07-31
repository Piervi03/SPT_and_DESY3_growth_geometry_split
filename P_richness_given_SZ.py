import numpy as np
from multiprocessing import Pool
from scipy.interpolate import interp1d
from scipy.special import ndtr

import scaling_relations


ln2pi = np.log(2.*np.pi)


def lnlike(catalog, SPT_survey_tab, HMF, cosmology, scaling, surveyCutRichness, richness_scatter_model, NPROC=0):
    """Returns ln-likelihood of all `catalog['richness']` measurements given
     `catalog['XI']`."""
    # Richness-mass relation, valid for all SPT fields
    lnrichness_z_m = scaling_relations.lnmass2lnobs('richness',
                                                    HMF['lnM_arr'][None, :], HMF['z_arr'][:, None],
                                                    scaling, cosmology)
    # Cycle through SPT fields (because that affects the zeta-mass relation)
    field_idx = np.nonzero([SPT_survey_tab['LAMBDA_MIN'][i] in ['deep', 'shallow']
                            for i in range(len(SPT_survey_tab))])[0]
    if NPROC == 0:
        lnlike_field = np.array([process_field(SPT_survey_tab[i],
                                               catalog['SPT_ID', 'XI', 'richness', 'REDSHIFT', 'FIELD'],
                                               surveyCutRichness, richness_scatter_model,
                                               HMF['z_arr'], HMF['lnM_arr'], HMF['richness_SZ_lndNdlnM'],
                                               lnrichness_z_m,
                                               scaling, cosmology)
                                 for i in field_idx])
    else:
        with Pool(processes=NPROC) as pool:
            lnlike_field = pool.starmap(process_field,
                                        [(SPT_survey_tab[i],
                                          catalog['XI', 'richness', 'REDSHIFT', 'FIELD'],
                                          surveyCutRichness, richness_scatter_model,
                                          HMF['z_arr'], HMF['lnM_arr'], HMF['richness_SZ_lndNdlnM'],
                                          lnrichness_m_interp,
                                          scaling, cosmology)
                                         for i in field_idx])
    lnlike = np.sum(lnlike_field)
    return lnlike


def process_field(SPT_field, catalog, surveyCutRichness, richness_scatter_model,
                  z_arr, lnM_arr, lndN_dz_dlnrichness_dlnzeta,
                  lnrichness_z_m,
                  scaling, cosmology):
    """Returns ln-likelihood of `catalog['richness']` given `catalog['XI']` for
    all clusters in `SPT_field`."""
    lnzeta_z_m = scaling_relations.lnmass2lnobs('zeta', lnM_arr[None, :], z_arr[:, None],
                                                scaling, cosmology,
                                                SPTfield=SPT_field)
    xi_z_m = scaling_relations.zeta2xi(np.exp(lnzeta_z_m))
    lambda_min = surveyCutRichness[SPT_field['LAMBDA_MIN']]
    lnlike = 0.
    field_idx = ((catalog['FIELD'] == SPT_field['FIELD']) & (catalog['richness'] > 0.)).nonzero()[0]
    for clusterID in field_idx:
        # Look up dN/dlnlambda/dlnzeta and observables at cluster redshift
        z_idx = np.argmin(np.abs(z_arr - catalog['REDSHIFT'][clusterID]))
        lndN_dlnrichness_dlnzeta = (lndN_dz_dlnrichness_dlnzeta[z_idx, :, :]
                                    + np.log(scaling_relations.dlnM_dlnobs('richness', scaling, z=catalog['REDSHIFT'][clusterID])
                                             * scaling_relations.dlnM_dlnobs('zeta', scaling)))
        lnrichness = lnrichness_z_m[z_idx, :]
        lnzeta = lnzeta_z_m[z_idx, :]
        xi = xi_z_m[z_idx, :]
        # Interpolate to fine xi array w/ Delta xi = 0.25 from max(xi-5, xi_min) to xi+3
        xi_min = np.amax([scaling_relations.zeta2xi(scaling['zeta_min']), catalog['XI'][clusterID] - 5.])
        xi_arr = np.arange(xi_min, catalog['XI'][clusterID] + 3., .25)
        lnzeta_arr = np.log(scaling_relations.xi2zeta(xi_arr))
        with np.errstate(invalid='ignore'):
            lndN_dlnrichness_dlnzeta_arr = interp1d(lnzeta, lndN_dlnrichness_dlnzeta, axis=1, assume_sorted=True)(lnzeta_arr)
        lndN_dlnrichness_dlnzeta_arr[np.isnan(lndN_dlnrichness_dlnzeta_arr)] = -np.inf
        # Condition on measured xi to get dN/dlnrichness
        lnP_xi = -.5 * (catalog['XI'][clusterID] - xi_arr)**2.
        lnitg = lnP_xi + lndN_dlnrichness_dlnzeta_arr
        with np.errstate(divide='ignore'):
            lndN_dlnrichness = np.log(np.sum(np.exp(.5 * (lnitg[:, :-1] + lnitg[:, 1:]))
                                      * (lnzeta_arr[1:] - lnzeta_arr[:-1]), axis=1))
        # Normalize to get dP/dlnrichness
        norm = np.sum(np.exp(.5*(lndN_dlnrichness[:-1] + lndN_dlnrichness[1:])) * (lnrichness[1:] - lnrichness[:-1]))
        lndN_dlnrichness -= np.log(norm)
        # P(richness_obs | richness), accounting for lambda_min
        richness = np.exp(lnrichness)
        if richness_scatter_model == 'lognormalrelPoisson':
            # var(ln richness) = 1/richness
            lnrichnessobs = np.log(catalog['richness'][clusterID])
            lndP_dobs = -.5 * (lnrichnessobs-lnrichness)**2*richness - .5*ln2pi + .5*lnrichness - lnrichnessobs
            with np.errstate(divide='ignore'):
                lndP_dobs -= np.log(1. - ndtr((np.log(lambda_min(catalog['REDSHIFT'][clusterID]))-lnrichness)*np.sqrt(richness)))
        elif richness_scatter_model == 'lognormalGaussPoisson':
            # var(richness) = richness
            lndP_dobs = -.5 * (catalog['richness'][clusterID]-richness)**2/richness - .5*ln2pi - .5*lnrichness
            with np.errstate(divide='ignore'):
                lndP_dobs -= np.log(1. - ndtr((lambda_min(catalog['REDSHIFT'][clusterID])-richness)/np.sqrt(richness)))
        lndP_dobs[np.isposinf(lndP_dobs)] = -np.inf
        lndP_dlnobs = lndP_dobs + lnrichness
        # ln-likelihood
        with np.errstate(invalid='ignore'):
            lnitg = lndP_dobs + lndN_dlnrichness
        this_lnlike = np.log(np.sum(np.exp(.5*(lnitg[:-1]+lnitg[1:])) * (lnrichness[1:]-lnrichness[:-1])))
        if not np.isfinite(this_lnlike):
            return -np.inf
        lnlike += this_lnlike
    return lnlike
