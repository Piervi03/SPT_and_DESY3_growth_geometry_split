import numpy as np
from multiprocessing import Pool
from scipy.special import log_ndtr, ndtr

import lensing, scaling_relations


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


def draw_SPTfield_gamma(N, rng, SPT_field):
    """Return area-weighted draws from SPT fields."""
    cum_area = np.cumsum(SPT_field['AREA'])
    cum_area /= cum_area[-1]
    field_idx = np.digitize(rng.random(N), cum_area) - 1
    field = SPT_field['FIELD'][field_idx]
    return field


def draw_lnobs_intrinsic_given_lnmass(rng, z, lnM, scaling, cosmology, covmat, SPT_field):
    """Return draws of ln[zeta, richness, DESWL] given `lnM`."""
    # Draw observable
    lnobs_draw_lnM = rng.multivariate_normal(lnM, covmat)
    # Convert to observable space
    lnrichness = scaling_relations.lnmass2lnobs('richness_base', lnobs_draw_lnM[1], z, scaling)
    lnMwl = scaling_relations.lnmass2lnobs('WLDES', lnobs_draw_lnM[2], z, scaling)
    lnzeta = scaling_relations.lnmass2lnobs('zeta', lnobs_draw_lnM[0], z, scaling, cosmology, SPTfield=SPT_field)
    return lnzeta, lnrichness, lnMwl


def draw_xi(rng, lnzeta):
    """Return draws of `xi` given `lnzeta`."""
    xi = rng.normal(scaling_relations.zeta2xi(np.exp(lnzeta)))
    return xi


def draw_richness_obs(rng, lnrichness, richness_scatter_model):
    """Return draws of observed richness given `lnrichness`."""
    if richness_scatter_model == 'lognormal':
        richness_obs = np.exp(lnrichness)
    elif richness_scatter_model == 'lognormalGaussPoisson':
        richness_int = np.exp(lnrichness)
        richness_obs = rng.normal(richness_int, np.sqrt(richness_int))
    elif richness_scatter_model == 'lognormalrelPoisson':
        richness_obs = np.exp(rng.normal(lnrichness, 1/np.sqrt(np.exp(lnrichness))))
    else:
        raise ValueError("Unknown richness scatter model: %s" % richness_scatter_model)
    return richness_obs


def lnP_greater_richnesscut(lnrichness, z, lambda_min_interp, richness_scatter_model):
    """Return ln-probability of exceeding `lambda_min` given `lnrichness`."""
    lambda_min = lambda_min_interp(z)
    if richness_scatter_model == 'lognormalGaussPoisson':
        richness = np.exp(lnrichness)
        lnP = log_ndtr((richness - lambda_min) / np.sqrt(richness))
    elif richness_scatter_model == 'lognormalrelPoisson':
        lnP = log_ndtr((lnrichness - np.log(lambda_min)) * np.sqrt(np.exp(lnrichness)))
    return lnP


def cov_lnM(scaling, z, lnM):
    """Return covariance matrix for [zeta, richness, DESWL] in mass space."""
    # Covariance in observable space
    scatter = np.array([scaling['Dsz'], scaling['Drichness'], 1.])
    corrmat = np.ones((3, 3))
    corrmat[0, 1] = scaling['rhoSZrichness']
    corrmat[0, 2] = scaling['rhoSZWL']
    corrmat[1, 2] = scaling['rhoWLrichness']
    corrmat[1, 0] = corrmat[0, 1]
    corrmat[2, 0] = corrmat[0, 2]
    corrmat[2, 1] = corrmat[1, 2]
    covmat = corrmat * scatter[:, None] * scatter[None, :]
    # Go to mass space
    dlnM_dlnobs = np.array([scaling_relations.dlnM_dlnobs(obs, scaling)
                            for obs in ['zeta', 'richness_base', 'WLDES']])
    covmat_lnM = covmat * dlnM_dlnobs[:, None] * dlnM_dlnobs[None, :] * np.ones((len(lnM), 3, 3))
    # Lensing scatter
    scatter = scaling_relations.WLscatter('WLDES', lnM, z, scaling)
    covmat_lnM[:, :, 2] *= scatter
    covmat_lnM[:, 2, :] *= scatter
    return covmat_lnM


def assign_SZrichness_bin(z, xi, richness, bins):
    """Return indices of observable bins."""


def assign_SZ_bin(z, xi, bins):



def get_mass_function_logprob(z, lnM, HMF_interp):
    """Return ln-probability of halo mass function
    ln(P(lnM)) = ln(dN/dlnM) at given `z` and array `lnM`."""
    # RectBivariateSpline wants sorted inputs, but this is faster than `grid=False`
    idx = np.argsort(lnM)
    lnprob = np.zeros(len(lnM))
    lnprob[idx] = HMF_interp(z, lnM[idx])
    return lnprob


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
