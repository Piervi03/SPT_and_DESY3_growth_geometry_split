import numpy as np
from scipy.interpolate import make_interp_spline
from cosmosis.datablock import option_section

import stacked_lnmass


def setup(options):
    SPT_survey_fields = options.get_string(option_section, 'SPT_survey_fields')
    config = {
              'NPROC': options.get_int(option_section, 'NPROC'),
              'SPT_survey_tab': np.genfromtxt(SPT_survey_fields, names=True, dtype=None),
              'z_bins': options.get_double_array_1d(option_section, 'SPTcl_z_bins'),
              'rot_mat': options.get_double_array_nd(option_section, 'rot_mat'),
              'rot_bins_x': options.get_double_array_1d(option_section, 'rot_bins_x'),
              'rot_bins_y': options.get_double_array_1d(option_section, 'rot_bins_y'),
              'richness_scatter_model': richness_scatter_model,
              }
    # lambda_min(z)
    surveyCutLambda_file = options.get_string(option_section, 'MCMF_lambda_min')
    tmp = np.genfromtxt(surveyCutLambda_file, names=True, dtype=None)
    config['survey_cut_richness'] = {}
    for name in tmp.dtype.names[1:]:
        config['survey_cut_richness'][name] = make_interp_spline(tmp['z'], tmp[name], k=1)
    return config


def execute(block, config):
    # Only need cosmo for E(z)-type stuff
    cosmology = {'Omega_m': block.get_double('cosmological_parameters', 'Omega_m'),
                 'Omega_l': block.get_double('cosmological_parameters', 'omega_lambda'),
                 'w0': block.get_double('cosmological_parameters', 'w'),
                 'wa': block.get_double('cosmological_parameters', 'wa')}
    # SZ scaling relation parameters
    scaling = {}
    for p in ['Asz', 'Bsz', 'Csz', 'Dsz', 'Esz', 'SPECS_calib', 'SZmPivot', 'zeta_min',
              'Arichness', 'Brichness', 'Crichness', 'Drichness', 'richmPivot',
              'Arichness_ext', 'Brichness_ext', 'Crichness_ext', 'Drichness_ext',
              'z_DESWISE']:
        scaling[p] = block.get_double('mor_parameters', p)
    # Halo mass function
    z, M, N = block.get_grid('HMF', 'z_arr', 'M_arr', 'dNdlnM')
    HMF = {'z_arr': z, 'lnM_arr': np.log(M), 'dNdlnM': N}
    # Compute the expected ln mass
    M = stacked_lnmass.execute(HMF,
                               cosmology, scaling,
                               config['SPT_survey_tab'],
                               config['survey_cut_richness'], config['richness_scatter_model'],
                               config['z_bins'], config['rot_mat'], config['rot_bins_x'], config['rot_bins_y'])
    if np.any(np.isnan(M)):
        return 1
    block.put_double_array_1d('mean_lnmass', 'M', M)
    return 0


def cleanup(config):
    pass
