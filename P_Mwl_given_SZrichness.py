import numpy as np
from multiprocessing import Pool
from scipy.special import log_ndtr, ndtr, ndtri
from scipy.interpolate import RectBivariateSpline

import lensing, scaling_relations


# At most, draw +3 sigma deviates
ndtr_max = ndtr(3.)


def execute(HMF, cosmology, scaling,
            SPT_survey_tab,
            survey_cut_richness, richness_scatter_model,
            N_draws=100000, NPROC=0):
    if NPROC == 0:
        z, xi, lnrichness, lnMwl, lnw = get_obs_draws(HMF,
                                                      cosmology, scaling,
                                                      SPT_survey_tab,
                                                      survey_cut_richness, richness_scatter_model,
                                                      N_draws=N_draws, seed=0)
    return z, xi, lnrichness, lnMwl, lnw


def get_obs_draws(HMF, cosmology, scaling,
                  SPT_survey_tab,
                  survey_cut_richness, richness_scatter_model,
                  N_draws=100000, seed=0):
    """Wrapper function that calls all workflow steps. Return observables."""
    # Initialize random number generator
    rng = np.random.default_rng(seed)
    # Set up halo mass function interpolation
    with np.errstate(divide='ignore'):
        HMF_interp = RectBivariateSpline(HMF['z_arr'], HMF['lnM_arr'], np.log(HMF['dNdlnM']), kx=1, ky=1)
    # Draw redshift and lnM
    z, lnM, lnw_HMF = draw_z_lnmass(rng, [.25, .95], [HMF['lnM_arr'][0], HMF['lnM_arr'][-1]], HMF_interp, N_draws)
    # Covariance matrix in lnM space
    covmat_lnM = cov_lnM(scaling, z, lnM)
    # Draw SPT field
    SPTfield = draw_SPTfield(N_draws, rng, SPT_survey_tab)
    # Draw ln[zeta, richness, DESWL] given lnM
    lnzeta, lnrichness, lnMwl, lnw_zeta = draw_lnobs_intrinsic_given_lnmass(rng,
                                                                            z, lnM,
                                                                            scaling, cosmology,
                                                                            covmat_lnM,
                                                                            SPTfield)
    # Draw xi given lnzeta
    xi, lnw_xi = draw_xi(rng, lnzeta, SPTfield)
    # Draw richness_obs given lnrichness
    richness_obs, lnw_richness = draw_richness_obs(rng, z, lnrichness,
                                                   survey_cut_richness, richness_scatter_model,
                                                   SPTfield)
    # Finalize
    lnw = lnw_HMF + lnw_zeta + lnw_xi + lnw_richness
    return z, xi, lnrichness, lnMwl, lnw


def draw_SPTfield(N, rng, SPT_field):
    """Return area-weighted draws from SPT fields."""
    MCMF_fields = ['_MCMF' in SPT_field['FIELD'][i]
                   for i in range(len(SPT_field))]
    cum_area = np.cumsum(SPT_field['AREA'][MCMF_fields])
    cum_area /= cum_area[-1]
    field_idx = np.digitize(rng.random(N), cum_area) - 1
    field = SPT_field[['FIELD', 'GAMMA', 'XI_MIN', 'LAMBDA_MIN', 'DELTA_CSZ']][MCMF_fields][field_idx]
    return field


def draw_lnobs_intrinsic_given_lnmass(rng, z, lnM, scaling, cosmology, covmat,
                                      SPT_field):
    """Return draws of ln[zeta, richness, DESWL] given `lnM`."""
    lnM_zetamin = np.array([scaling_relations.obs2lnmass('zeta', scaling['zeta_min'], z[i],
                                                         scaling, cosmology,
                                                         SPTfield=SPT_field[i])
                            for i in range(len(z))])
    # Draw zeta>zeta_min | lnM
    SZscatter_lnM = np.sqrt(covmat[:, 0, 0])
    ln_weight = log_ndtr((lnM_zetamin - lnM) / SZscatter_lnM)
    r_min = ndtr((lnM_zetamin - lnM) / SZscatter_lnM)
    r = r_min + (ndtr_max-r_min) * rng.random(len(lnM))
    lnM_zeta = lnM + ndtri(r) * SZscatter_lnM
    # Draw ln(richness, DESWL) | ln(zeta, lnM)
    mean_cond = lnM[:, None] + covmat[:, 0, 1:]/covmat[:, 0, 0][:, None] * (lnM_zeta - lnM)[:,None]
    var_cond = np.linalg.inv(np.linalg.inv(covmat)[:, 1:, 1:])
    # var_cond_ = covmat[:, 1:, 1:] - np.array([np.matmul(covmat[i, 1:, 0], covmat[i, 0, 1:]) for i in range(len(lnM))])[:, None, None] / covmat[:, 0, 0][:, None, None]
    lnobs_lnM = np.array([rng.multivariate_normal(mean_cond[i], var_cond[i])
                          for i in range(len(lnM))])
    # Observable space
    lnzeta = np.array([scaling_relations.lnmass2lnobs('zeta', lnM_zeta[i], z[i],
                                                      scaling, cosmology,
                                                      SPTfield=SPT_field[i])
                       for i in range(len(z))])
    lnrichness = scaling_relations.lnmass2lnobs('richness', lnobs_lnM[:, 0], z, scaling)
    lnMwl = scaling_relations.lnmass2lnobs('WLDES', lnobs_lnM[:, 1], z, scaling)
    return lnzeta, lnrichness, lnMwl, ln_weight


def draw_xi(rng, lnzeta, SPT_field):
    """Return draws of `xi` given `lnzeta`."""
    # Draw (xi>XI_MIN)|zeta
    xi_mean = scaling_relations.zeta2xi(np.exp(lnzeta))
    r_min = ndtr(SPT_field['XI_MIN'] - xi_mean)
    r = r_min + (ndtr_max-r_min) * rng.random(len(lnzeta))
    xi = xi_mean + ndtri(r)
    # Account for xi>XI_MIN
    lnw = np.log(1. - r_min)
    return xi, lnw


def draw_richness_obs(rng, z, lnrichness,
                      survey_cut_richness, richness_scatter_model,
                      SPT_field):
    """Return draws of observed richness given `lnrichness`, accounting for
    lambda_min(z)."""
    lambda_min = np.array([survey_cut_richness[SPT_field['LAMBDA_MIN'][i]](z[i])
                           for i in range(len(lnrichness))])
    richness = np.exp(lnrichness)
    if richness_scatter_model == 'lognormalGaussPoisson':
        # var(richness) = richness
        r_min = ndtr((lambda_min - richness) / np.sqrt(richness))
        r = r_min + (ndtr_max - r_min) * rng.random(len(lnrichness))
        richness_obs = richness + ndtri(r) * np.sqrt(richness)
        lnw = np.log(1. - r_min)
    elif richness_scatter_model == 'lognormalrelPoisson':
        lnlambda_min = np.log(lambda_min)
        # var(ln richness) = 1/richness
        r_min = ndtr((lnlambda_min - lnrichness) * np.sqrt(richness))
        r = r_min + (ndtr_max - r_min) * rng.random(len(lnrichness))
        richness_obs = np.exp(lnrichness + ndtri(r_min) / np.sqrt(richness))
        lnw = np.log(1. - r_min)
    else:
        raise ValueError("Unknown richness scatter model: %s" % richness_scatter_model)
    return richness_obs, lnw


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
    covmat_lnM = (covmat * dlnM_dlnobs[:, None] * dlnM_dlnobs[None, :])[None, :, :] * np.ones((len(lnM), 3, 3))
    # Lensing scatter
    scatter = scaling_relations.WLscatter('WLDES', lnM, z, scaling)
    covmat_lnM[:, :, 2] *= scatter[:, None]
    covmat_lnM[:, 2, :] *= scatter[:, None]
    return covmat_lnM


def draw_z_lnmass(rng, z_lim, lnM_lim, HMF_interp, N_draws):
    """Return draws of `z` and `lnM`, and ln-probability of halo mass function
    ln(P(lnM)) = ln(dN/dlnM) at `z` and `lnM`."""
    # Uniform draws of redshift and lnM
    z = rng.uniform(z_lim[0], z_lim[1], size=N_draws)
    lnM = rng.uniform(lnM_lim[0], lnM_lim[1], size=N_draws)
    lnprob = HMF_interp(z, lnM, grid=False)
    return z, lnM, lnprob
