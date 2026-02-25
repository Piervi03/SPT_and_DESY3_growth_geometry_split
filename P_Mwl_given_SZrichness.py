import numpy as np
# from multiprocessing import Pool
from scipy.special import log_ndtr, ndtr, ndtri
from scipy.interpolate import RectBivariateSpline

import scaling_relations


# At most, draw +4 sigma deviates
ndtr_max = ndtr(4.)


def execute(HMF, cosmology, scaling,
            SPT_survey_tab,
            survey_cut_richness, richness_scatter_model,
            z_lims,
            N_draws=1000000, NPROC=0):
    if NPROC == 0:
        z, xi, SNR, richness, lnMwl, lnw = get_obs_draws(HMF,
                                                         cosmology, scaling,
                                                         SPT_survey_tab,
                                                         survey_cut_richness, richness_scatter_model,
                                                         z_lims,
                                                         N_draws=N_draws, seed=0)
    else:
        raise Exception("Multiprocessing not implemented yet.")
    lnw -= np.amax(lnw)
    return z, xi, SNR, richness, lnMwl, lnw


def get_obs_draws(HMF, cosmology, scaling,
                  SPT_survey_tab,
                  survey_cut_richness, richness_scatter_model,
                  z_lims,
                  N_draws=1000000, seed=0):
    """Wrapper function that calls all workflow steps. Return observables and
    ln-weights."""
    # Initialize random number generator
    rng = np.random.default_rng(seed)
    # Set up halo mass function interpolation
    with np.errstate(divide='ignore'):
        HMF_interp = RectBivariateSpline(HMF['z_arr'], HMF['lnM_arr'], np.log(HMF['dNdlnM']), kx=1, ky=1)
    # Draw redshift and lnM
    z, lnM, lnw_HMF = draw_z_lnmass(rng, z_lims, [HMF['lnM_arr'][0], HMF['lnM_arr'][-1]], HMF_interp, N_draws)
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
    xi, SNR, lnw_xi = draw_xi(rng, lnzeta, SPTfield)
    lnw = lnw_HMF + lnw_zeta + lnw_xi
    # Only keep valid draws
    idx = np.isfinite(lnw)
    if not np.all(idx):
        lnw = lnw[idx]
        z = z[idx]
        lnMwl = lnMwl[idx]
        xi = xi[idx]
        SNR = SNR[idx]
        lnrichness = lnrichness[idx]
        if len(SPTfield) > 1:
            SPTfield = SPTfield[idx]
    # Draw richness_obs given lnrichness
    richness_obs, lnw_richness = draw_richness_obs(rng, z, lnrichness,
                                                   survey_cut_richness, richness_scatter_model,
                                                   SPTfield)
    lnw += lnw_richness
    # Only keep valid draws
    idx = np.isfinite(lnw)
    if not np.all(idx):
        z = z[idx]
        xi = xi[idx]
        SNR = SNR[idx]
        richness_obs = richness_obs[idx]
        lnMwl = lnMwl[idx]
        lnw = lnw[idx]
    return z, xi, SNR, richness_obs, lnMwl, lnw


def draw_SPTfield(N, rng, SPT_field):
    """Return area-weighted draws from SPT fields. If there is only one field,
    then this is trivial."""
    # Only one field
    if len(SPT_field) == 1:
        field = SPT_field[['FIELD', 'GAMMA', 'XI_MIN', 'LAMBDA_MIN', 'DELTA_CSZ']]
    # Multiple fields
    else:
        MCMF_fields = np.nonzero([SPT_field['LAMBDA_MIN'][i] not in ['none', 'None', 'NONE']
                                  for i in range(len(SPT_field))])[0]
        if len(MCMF_fields) == 1:
            # No need to draw fields
            field = SPT_field[['FIELD', 'GAMMA', 'XI_MIN', 'LAMBDA_MIN', 'DELTA_CSZ']][MCMF_fields]
        else:
            # Area-weighted draws
            cum_area = np.cumsum(SPT_field['AREA'][MCMF_fields])
            cum_area /= cum_area[-1]
            field_idx = np.digitize(rng.random(N), cum_area) - 1
            field = SPT_field[['FIELD', 'GAMMA', 'XI_MIN', 'LAMBDA_MIN', 'DELTA_CSZ']][MCMF_fields][field_idx]
    return field


def draw_lnobs_intrinsic_given_lnmass(rng, z, lnM, scaling, cosmology, covmat,
                                      SPT_field):
    """Return draws of ln[zeta, richness, DESWL] given `lnM`."""
    lnM_zetamin = scaling_relations.obs2lnmass('zeta', scaling['zeta_min'], z,
                                               scaling, cosmology,
                                               SPTfield=SPT_field)
    # Draw zeta>zeta_min | lnM
    SZscatter_lnM = np.sqrt(covmat[:, 0, 0])
    ln_weight = log_ndtr((lnM - lnM_zetamin) / SZscatter_lnM)
    r_min = ndtr((lnM_zetamin - lnM) / SZscatter_lnM)
    r = r_min + (ndtr_max-r_min) * rng.random(len(lnM))
    lnM_zeta = lnM + ndtri(r) * SZscatter_lnM
    ln_weight[r_min > ndtr_max] = -np.inf
    # Draw ln(richness, DESWL) | ln(zeta, lnM)
    mean_cond = lnM[:, None] + covmat[:, 0, 1:]/covmat[:, 0, 0][:, None] * (lnM_zeta - lnM)[:, None]
    var_cond = np.linalg.inv(np.linalg.inv(covmat)[:, 1:, 1:])
    # rng.multivariate_normal only accepts single mean and cov, cholesky broadcasts
    # and is faster than svd. Gain factor ~30 in speed.
    std_normal_draws = rng.standard_normal((len(lnM), 2))
    cho = np.linalg.cholesky(var_cond).transpose((0, 2, 1))
    lnobs_lnM = mean_cond + np.matmul(std_normal_draws[:, None, :], cho)[:, 0, :]
    # Observable space
    lnzeta = scaling_relations.lnmass2lnobs('zeta', lnM_zeta, z,
                                            scaling, cosmology,
                                            SPTfield=SPT_field)
    lnrichness = scaling_relations.lnmass2lnobs('richness', lnobs_lnM[:, 0], z, scaling)
    lnMwl = scaling_relations.lnmass2lnobs('WLDES', lnobs_lnM[:, 1], z, scaling)
    return lnzeta, lnrichness, lnMwl, ln_weight


def draw_xi(rng, lnzeta, SPT_field):
    """Return draws of `xi` given `lnzeta`."""
    # Draw xi>XI_MIN | zeta
    xi_mean = scaling_relations.zeta2xi(np.exp(lnzeta))
    r_min = ndtr(SPT_field['XI_MIN'] - xi_mean)
    r = r_min + (ndtr_max-r_min) * rng.random(len(lnzeta))
    xi = xi_mean + ndtri(r)
    # Account for xi>XI_MIN
    lnw = log_ndtr(xi_mean-SPT_field['XI_MIN'])
    # Catch cases where xi_mean + 4 < XI_MIN
    lnw[r_min > ndtr_max] = -np.inf
    # For binning
    SNR = scaling_relations.xi2zeta(xi)/SPT_field['GAMMA']
    return xi, SNR, lnw


def draw_richness_obs(rng, z, lnrichness,
                      survey_cut_richness, richness_scatter_model,
                      SPT_field):
    """Return draws of observed richness given `lnrichness`, accounting for
    lambda_min(z)."""
    # Lambda_min(z) is a function of redshift and SPT field
    if len(SPT_field) == 1:
        lambda_min = survey_cut_richness[SPT_field['LAMBDA_MIN'][0]](z)
    else:
        lambda_min = np.array([survey_cut_richness[SPT_field['LAMBDA_MIN'][i]](z[i])
                               for i in range(len(z))])
    # Draw richness_obs given lnrichness
    richness = np.exp(lnrichness)
    if richness_scatter_model in ['lognormalGaussPoisson', 'lognormalGausssuperPoisson']:
        if richness_scatter_model == 'lognormalGaussPoisson':
            richness_err = np.sqrt(richness)
        else:
            richness_err = np.sqrt(richness+10.) * (1.08 + .45*(z-.6))
        # var(richness) = richness
        r_min = ndtr((lambda_min - richness) / richness_err)
        r = r_min + (ndtr_max - r_min) * rng.random(len(lnrichness))
        richness_obs = richness + ndtri(r) * richness_err
        # Account for lambda_min
        lnw = log_ndtr((richness - lambda_min) / richness_err)
        lnw[r_min > ndtr_max] = -np.inf
    elif richness_scatter_model == 'lognormalrelPoisson':
        lnlambda_min = np.log(lambda_min)
        # var(ln richness) = 1/richness
        r_min = ndtr((lnlambda_min - lnrichness) * np.sqrt(richness))
        r = r_min + (ndtr_max - r_min) * rng.random(len(lnrichness))
        richness_obs = np.exp(lnrichness + ndtri(r_min) / np.sqrt(richness))
        # Account for lambda_min
        lnw = log_ndtr((lnrichness - lnlambda_min) * np.sqrt(richness))
        lnw[r_min > ndtr_max] = -np.inf
    else:
        raise ValueError("Unknown richness scatter model: %s" % richness_scatter_model)
    return richness_obs, lnw


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
