import numpy as np
from multiprocessing import Pool
from scipy.interpolate import make_interp_spline
import pickle

import scaling_relations


def lnlike(catalog, SPT_survey_tab, HMF, cosmology, scaling, surveyCutRichness, NPROC=0):
    """Returns ln-likelihood of all `catalog['richness']` measurements given
     `catalog['XI']`."""
    # Richness-mass relation, valid for all SPT fields
    lnrichness_m = scaling_relations.lnmass2lnobs('richness',
                                                  HMF['lnM_arr'][None, :], HMF['z_arr'][:, None],
                                                  scaling, cosmology)
    lnrichness_m_interp = make_interp_spline(HMF['z_arr'], lnrichness_m, axis=0, k=1)
    # Cycle through SPT fields (because that affects the zeta-mass relation)
    if NPROC == 0:
        lnlike_field = np.array([process_field(SPT_survey_tab[i],
                                               catalog['SPT_ID', 'XI', 'richness', 'REDSHIFT', 'FIELD'],
                                               surveyCutRichness,
                                               HMF['z_arr'], HMF['lnM_arr'], HMF['richness_SZ_lndNdlnM'],
                                               lnrichness_m_interp,
                                               scaling, cosmology)
                                 for i in range(len(SPT_survey_tab))])
    else:
        with Pool(processes=NPROC) as pool:
            lnlike_field = pool.starmap(process_field,
                                        [(SPT_survey_tab[i],
                                          catalog['XI', 'richness', 'REDSHIFT', 'FIELD'],
                                          surveyCutRichness,
                                          HMF['z_arr'], HMF['lnM_arr'], HMF['richness_SZ_lndNdlnM'],
                                          lnrichness_m_interp,
                                          scaling, cosmology)
                                         for i in range(len(SPT_survey_tab))])
    lnlike = np.sum(lnlike_field)
    return lnlike


def process_field(SPT_field, catalog, surveyCutRichness,
                  z_arr, lnM_arr, lndN_dz_dlnrichness_dlnzeta,
                  lnrichness_m_interp,
                  scaling, cosmology):
    """Returns ln-likelihood of `catalog['richness']` given `catalog['XI']` for
    all clusters in `SPT_field`."""
    if SPT_field['FIELD'] == 'sptpol_500d_MCMF':
        lambda_min = surveyCutRichness['deep']
    else:
        lambda_min = surveyCutRichness['shallow']
    lnlike = 0.
    field_idx = ((catalog['FIELD'] == SPT_field['FIELD']) & (catalog['richness'] > 0.)).nonzero()[0]
    for clusterID in field_idx:
        # Look up dN/dlnlambda/dlnzeta and richness at cluster redshift
        lndN_dlnrichness_dlnzeta = (lndN_dz_dlnrichness_dlnzeta[np.argmin(np.abs(z_arr - catalog['REDSHIFT'][clusterID])), :, :]
                                    + np.log(scaling_relations.dlnM_dlnobs('richness', scaling, z=catalog['REDSHIFT'][clusterID])
                                             * scaling_relations.dlnM_dlnobs('zeta', scaling)))
        richness = np.exp(lnrichness_m_interp(catalog['REDSHIFT'][clusterID]))
        # zeta and xi
        lnzeta = scaling_relations.lnmass2lnobs('zeta', lnM_arr, catalog['REDSHIFT'][clusterID],
                                                scaling, cosmology,
                                                SPTfield=SPT_field)
        xi = scaling_relations.zeta2xi(np.exp(lnzeta))
        # Integrate over xi from max(xi-5, xi_min) to xi+3
        xi_min = np.amax([scaling_relations.zeta2xi(scaling['zeta_min']), catalog['XI'][clusterID] - 5.])
        idx = (xi > xi_min) & (xi < catalog['XI'][clusterID] + 3.)
        lnP_xi = -.5 * (catalog['XI'][clusterID] - xi[idx])**2.
        lnitg = lnP_xi + lndN_dlnrichness_dlnzeta[:, idx]
        P_lambda = np.sum(np.exp(.5 * (lnitg[:, :-1] + lnitg[:, 1:]))
                          * (lnzeta[idx][1:] - lnzeta[idx][:-1]), axis=1) / richness
        # Richness ln-likelihood, allowing for extrapolation in ln(P)
        idx = P_lambda > 0.
        if len(idx.nonzero()[0]) < 3:
            return -np.inf
        lnP_lambda_interp = make_interp_spline(richness[idx], np.log(P_lambda[idx]))
        P_lambda_interp = make_interp_spline(richness, P_lambda)
        this_lnlike = (lnP_lambda_interp(catalog['richness'][clusterID])
                       - np.log(P_lambda_interp.integrate(lambda_min(catalog['REDSHIFT'][clusterID]), richness[-1])))
        # Finalize
        if not np.isfinite(this_lnlike):
            return -np.inf
        lnlike += this_lnlike
    return lnlike
