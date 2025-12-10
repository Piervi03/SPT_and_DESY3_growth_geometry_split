import numpy as np
from multiprocessing import Pool
from scipy.integrate import simpson
from scipy.interpolate import RectBivariateSpline
from scipy.ndimage import gaussian_filter1d

import cosmo
import scaling_relations

sqrt2pi = np.sqrt(2.*np.pi)
Delta_xi = .2


def run(HMF, cosmology, scaling, **kwargs):
    """Return ln-likelihood for SPT cluster abundance."""
    SPT_survey = kwargs.pop('SPT_survey')
    surveyCutSZmax = kwargs.pop('surveyCutSZmax')
    z_cl_min_max = kwargs.pop('z_cl_min_max')
    richness_scatter_model = kwargs.pop('richness_scatter_model')
    lambda_min = kwargs.pop('lambda_min')
    lambda_out = kwargs.pop('lambda_out')
    SNR_red_out = kwargs.pop('SNR_red_out')
    NPROC = kwargs.pop('NPROC', 0)
    # Compute distribution for each SPT field (optional multiprocessing)
    num_fields = len(SPT_survey)
    if NPROC == 0:
        field_results = [run_field(SPT_survey[i],
                                   surveyCutSZmax, z_cl_min_max,
                                   richness_scatter_model, lambda_min,
                                   lambda_out, SNR_red_out,
                                   HMF, cosmology, scaling) for i in range(num_fields)]
    else:
        with Pool(processes=NPROC) as pool:
            field_results = pool.starmap(run_field,
                                         [(SPT_survey[i],
                                           surveyCutSZmax, z_cl_min_max,
                                           richness_scatter_model, lambda_min,
                                           lambda_out, SNR_red_out,
                                           HMF, cosmology, scaling)
                                          for i in range(num_fields)])
    res = np.array([field_results[i] for i in range(num_fields)]).sum(axis=0)
    return res


def run_field(SPTfield,
              surveyCutSZmax, z_cl_min_max,
              richness_scatter_model, lambda_min,
              lambda_out, SNR_red_out,
              HMF, cosmology, scaling):
    """Return dN/dSNR/dlambda for a given `SPTfield`."""
    z_idx = ((HMF['z_arr'] >= z_cl_min_max[0]) & (HMF['z_arr'] <= z_cl_min_max[1])).nonzero()[0]
    # Survey field
    if SPTfield['LAMBDA_MIN'] in ['none', 'None', 'NONE']:
        return np.zeros((len(lambda_out), len(SNR_red_out)))
    lnzeta_m = scaling_relations.lnmass2lnobs('zeta', HMF['lnM_arr'][None, :], HMF['z_arr'][z_idx, None],
                                              scaling, cosmology, SPTfield=SPTfield)
    lnrichness_m = scaling_relations.lnmass2lnobs('richness', HMF['lnM_arr'][None, :], HMF['z_arr'][z_idx, None],
                                                  scaling)
    # dN/dlnlambda/dlnzeta
    dN_dz_dlnlambda_dlnzeta = (np.exp(HMF['richness_SZ_lndNdlnM'][z_idx])
                               * scaling_relations.dlnM_dlnobs('zeta', scaling)
                               * scaling_relations.dlnM_dlnobs('richness', scaling, z=HMF['z_arr'][z_idx])[:, None, None]
                               * (np.pi/180)**2 * SPTfield['AREA'])
    # Cut in zeta
    dN_dz_dlnlambda_dlnzeta[lnzeta_m[:, None, :]*np.ones(dN_dz_dlnlambda_dlnzeta.shape) < np.log(scaling['zeta_min'])] = 0.
    # dN/dlnlambda/dxi = dN/dlnlambda/dlnzeta * dlnzeta/dxi
    dN_dz_dlnlambda_dxi = dN_dz_dlnlambda_dlnzeta * scaling_relations.dlnzeta_dxi_given_zeta(np.exp(lnzeta_m))[:, None, :]
    # Interpolate to regular xi grid
    xi_zeta = scaling_relations.zeta2xi(np.exp(lnzeta_m))
    xi_bins = np.arange(scaling_relations.zeta2xi(scaling['zeta_min']), surveyCutSZmax, Delta_xi)
    dN_dz_dlnlambda_dxi_bins = np.array([RectBivariateSpline(HMF['lnM_arr'], xi_zeta[i, :],
                                                             dN_dz_dlnlambda_dxi[i, :, :], kx=1, ky=1)(HMF['lnM_arr'], xi_bins)
                                         for i in range(len(z_idx))])
    # Convolve with standard normal in xi
    dN_dz_dlnlambda_dxi_conv = gaussian_filter1d(dN_dz_dlnlambda_dxi_bins, sigma=1./Delta_xi, axis=-1, mode='constant')
    xi_bins = xi_bins[::2]
    dN_dz_dlnlambda_dxi_conv = dN_dz_dlnlambda_dxi_conv[:, :, ::2]
    # Convolve with measurement error in lambda
    if richness_scatter_model in ['lognormalGaussPoisson', 'lognormalGausssuperPoisson']:
        richness = np.exp(lnrichness_m)
        # Obs scatter [z, richness_int]
        if richness_scatter_model == 'lognormalGaussPoisson':
            std = np.sqrt(richness)
        else:
            std = np.sqrt(richness+10) * (1.08 + .45*(HMF['z_arr'][z_idx, None]-.6))
        # integrand shape [z, richness, richness_int, xi]
        # P(richness | richness_int) [z, richness, richness_int]
        Prich_gvn_richint = (np.exp(-.5 * ((richness[:, :, None] - richness[:, None, :])/std[:, None, :])**2)
                             / (sqrt2pi * std[:, None, :]))
        # integrand P(richness | richness_int) d3N/dz/drichness_int/dxi [z, richness, richness_int, xi]
        dN_dz_dlambda_dxi = np.array([simpson(dN_dz_dlnlambda_dxi_conv[i, None, :, :] * Prich_gvn_richint[i, :, :, None],
                                              lnrichness_m[i], axis=1)
                                      for i in range(len(z_idx))])
    else:
        raise ValueError("richness_scatter_model {} is not supported".format(richness_scatter_model))
    # Apply lambda_min(z)
    lambda_min_z = lambda_min[SPTfield['LAMBDA_MIN']](HMF['z_arr'][z_idx])
    for i in range(len(z_idx)):
        dN_dz_dlambda_dxi[i, richness[i] < lambda_min_z[i], :] = 0.
    # Apply XI_MIN
    dN_dz_dlambda_dxi[:, :, xi_bins < SPTfield['XI_MIN']] = 0.
    # Rescale to account for redshift dependence
    dN_dz_dlambda_dxi_scaled = np.empty((len(z_idx), len(lambda_out), len(SNR_red_out)))
    lnzeta_bins = np.log(scaling_relations.xi2zeta(xi_bins))
    for i, z in enumerate(HMF['z_arr'][z_idx]):
        lnrichness_scaled = np.log(lambda_out * (1+z)**scaling['Crichness'])
        lnzeta_scaled = np.log(SNR_red_out * SPTfield['GAMMA']
                               * cosmo.Ez(z, cosmology)**scaling['Csz'])
        dN_dz_dlambda_dxi_scaled[i] = RectBivariateSpline(lnrichness_m[i], lnzeta_bins,
                                                          dN_dz_dlambda_dxi[i], kx=1, ky=1)(lnrichness_scaled, lnzeta_scaled)
    # Integrate over redshift
    dN_dlambda_dxi_scaled = simpson(dN_dz_dlambda_dxi_scaled, HMF['z_arr'][z_idx], axis=0)
    return dN_dlambda_dxi_scaled
