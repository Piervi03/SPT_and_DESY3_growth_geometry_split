import numpy as np
from scipy.integrate import simpson
from multiprocessing import Pool
from scipy.interpolate import interp1d, make_interp_spline
from scipy.special import ndtr

import scaling_relations


sqrt2pi = np.sqrt(2.*np.pi)
ln2pi = np.log(2.*np.pi)


def lnlike(catalog, SPT_survey_tab, HMF, cosmology, scaling, lambda_min, richness_scatter_model, NPROC=0):
    """Returns ln-likelihood of all `catalog['richness']` measurements given
     `catalog['XI']`."""
    # Richness-mass relation, valid for all SPT fields
    lnrichness_z_m = scaling_relations.lnmass2lnobs('richness',
                                                    HMF['lnM_arr'][None, :], HMF['z_arr'][:, None],
                                                    scaling, cosmology)
    # Only fields with lambda_min have richness measurements
    field_idx = np.nonzero([SPT_survey_tab['LAMBDA_MIN'][i] not in ['None', 'none', 'NONE']
                            for i in range(len(SPT_survey_tab))])[0]
    # Each field separately because zeta-mass relation changes
    if NPROC == 0:
        lnlike_field = np.array([process_field(SPT_survey_tab[i],
                                               catalog['SPT_ID', 'XI', 'richness', 'REDSHIFT', 'FIELD'],
                                               lambda_min, richness_scatter_model,
                                               HMF['z_arr'], HMF['lnM_arr'], HMF['richness_SZ_lndNdlnM'],
                                               lnrichness_z_m,
                                               scaling, cosmology)
                                 for i in field_idx])
    else:
        with Pool(processes=NPROC) as pool:
            lnlike_field = pool.starmap(process_field,
                                        [(SPT_survey_tab[i],
                                          catalog['XI', 'richness', 'REDSHIFT', 'FIELD'],
                                          lambda_min, richness_scatter_model,
                                          HMF['z_arr'], HMF['lnM_arr'], HMF['richness_SZ_lndNdlnM'],
                                          lnrichness_z_m,
                                          scaling, cosmology)
                                         for i in field_idx])
    lnlike = np.sum(lnlike_field)
    return lnlike


def process_field(SPT_field, catalog, lambda_min, richness_scatter_model,
                  z_arr, lnM_arr, lndN_dz_dlnrichness_dlnzeta_z,
                  lnrichness_z_m,
                  scaling, cosmology):
    """Returns ln-likelihood of `catalog['richness']` given `catalog['XI']` at
    `catalog['REDSHIFT']` for all clusters in `SPT_field`."""
    lnzeta_z_m = scaling_relations.lnmass2lnobs('zeta', lnM_arr[None, :], z_arr[:, None],
                                                scaling, cosmology,
                                                SPTfield=SPT_field)
    lnlike = 0.
    field_idx = ((catalog['FIELD'] == SPT_field['FIELD']) & (catalog['richness'] > 0.)).nonzero()[0]
    for clusterID in field_idx:
        # Look up dN(z)/dlnlambda/dlnzeta and observables at cluster redshift
        z_idx = np.argmin(np.abs(z_arr - catalog['REDSHIFT'][clusterID]))
        # With constant slope, we ignore the constant factor and assume dN/lnobs = dN/lnM
        lndN_dlnrichness_dlnzeta = lndN_dz_dlnrichness_dlnzeta_z[z_idx, :, :]
        lnrichness = lnrichness_z_m[z_idx, :]
        lnzeta = lnzeta_z_m[z_idx, :]
        this_lambda_min = lambda_min[SPT_field['LAMBDA_MIN']](catalog['REDSHIFT'][clusterID])
        # Interpolate to fine xi array w/ Delta xi = 0.25 from max(xi-4, xi_min) to xi+4
        xi_min = np.amax([scaling_relations.zeta2xi(scaling['zeta_min']), catalog['XI'][clusterID] - 4.])
        xi_arr = np.arange(xi_min, catalog['XI'][clusterID] + 4., .25)
        lnzeta_arr = np.log(scaling_relations.xi2zeta(xi_arr))
        with np.errstate(invalid='ignore'):
            lndN_dlnrichness_dlnzeta_arr = interp1d(lnzeta, lndN_dlnrichness_dlnzeta,
                                                    axis=1, kind='linear',
                                                    assume_sorted=True)(lnzeta_arr)
        lndN_dlnrichness_dlnzeta_arr[np.isnan(lndN_dlnrichness_dlnzeta_arr)] = -np.inf
        # Condition dN/dlnrichness/dlnzeta (lnitg) on measured xi
        itg = np.exp(lndN_dlnrichness_dlnzeta_arr - .5 * (catalog['XI'][clusterID] - xi_arr)**2.)
        # Marginalize over zeta to get dN/dlnrichness
        dN_dlnrichness = simpson(itg, lnzeta_arr, axis=1)
        # No observational scatter in richness (or rather, absorbed in intrinsic scatter)
        if richness_scatter_model == 'lognormal':
            if this_lambda_min == 0.:
                # Normalize to get dP/dlnrichness
                dN_dlnrichness /= simpson(dN_dlnrichness, lnrichness)
                this_lnlike = make_interp_spline(lnrichness, np.log(dN_dlnrichness), k=2)(
                                                 np.log(catalog['richness'][clusterID]))
            else:
                # Insert lambda_min into lnrichness array
                lnthis_lambda_min = np.log(this_lambda_min)
                lnrichness_cut = lnrichness[lnrichness > lnthis_lambda_min]
                lnrichness_cut = np.insert(lnrichness_cut, 0, lnthis_lambda_min)
                lndN_dlnrichness_cut = make_interp_spline(lnrichness, np.log(dN_dlnrichness), k=2)(lnrichness_cut)
                # Normalize to get dP/dlnrichness
                lndN_dlnrichness_cut -= np.log(simpson(np.exp(lndN_dlnrichness_cut), lnrichness_cut))
                this_lnlike = make_interp_spline(lnrichness_cut, lndN_dlnrichness_cut, k=2)(
                                                 np.log(catalog['richness'][clusterID]))
        # Models with observational scatter in richness
        elif richness_scatter_model in ['lognormalrelPoisson', 'lognormalGaussPoisson', 'lognormalGausssuperPoisson']:
            # Normalize to get dP/dlnrichness
            dN_dlnrichness /= simpson(dN_dlnrichness, lnrichness)
            # P(richness_obs | richness) accounting for lambda_min
            richness = np.exp(lnrichness)
            if richness_scatter_model == 'lognormalrelPoisson':
                # var(ln richness) = 1/richness
                lnrichnessobs = np.log(catalog['richness'][clusterID])
                dP_dobs = (np.exp(-.5 * (lnrichnessobs-lnrichness)**2*richness)
                           / np.sqrt(2. * np.pi * catalog['richness'][clusterID] / richness))
                if this_lambda_min > 0.:
                    with np.errstate(divide='ignore'):
                        dP_dobs /= (1. - ndtr((np.log(this_lambda_min)-lnrichness)*np.sqrt(richness)))
            elif richness_scatter_model in ['lognormalGaussPoisson', 'lognormalGausssuperPoisson']:
                if richness_scatter_model == 'lognormalGaussPoisson':
                    std_richness = np.sqrt(richness)
                elif richness_scatter_model == 'lognormalGausssuperPoisson':
                    std_richness = np.sqrt(richness+10.) * (1.08 + .45*(catalog['REDSHIFT'][clusterID]-.6))
                dP_dobs = np.exp(-.5 * ((catalog['richness'][clusterID]-richness)/std_richness)**2) / (sqrt2pi * std_richness)
                with np.errstate(invalid='ignore', divide='ignore'):
                    dP_dobs /= (1. - ndtr((this_lambda_min-richness)/std_richness))
            # Likelihood = int dlambda P(lambda_obs|lambda) P(lambda)
            with np.errstate(invalid='ignore'):
                itg = dP_dobs * dN_dlnrichness
            # Remove infs and set up interpolation
            finite = np.isfinite(itg)
            lnrichness = lnrichness[finite]
            itg_interp = make_interp_spline(lnrichness, itg[finite], k=2)
            this_lnlike = np.log(itg_interp.integrate(np.amin(lnrichness), np.amax(lnrichness)))
        else:
            raise ValueError('Invalid richness_scatter_model {}'.format(richness_scatter_model))
        if not np.isfinite(this_lnlike):
            return -np.inf
        lnlike += this_lnlike
    return lnlike
