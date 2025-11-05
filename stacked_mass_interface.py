import numpy as np
from scipy.interpolate import make_interp_spline
from astropy.table import Table
import h5py
from cosmosis.datablock import option_section

import stacked_mass


def setup(options):
    SPT_survey_fields = options.get_string(option_section, 'SPT_survey_fields')
    config = {
              'NPROC': options.get_int(option_section, 'NPROC', 0),
              'SPT_survey_tab': Table.read(SPT_survey_fields, format='ascii.commented_header'),
              'z_bins': options.get_double_array_1d(option_section, 'z_bins'),
              'rot_bins_x': options.get_double_array_1d(option_section, 'rot_bins_x'),
              'rot_bins_y': options.get_double_array_1d(option_section, 'rot_bins_y'),
              'richness_scatter_model': options.get_string(option_section, 'richness_scatter_model'),
              }
    # Observable rotation matrix
    rot_mat_row = options.get_double_array_1d(option_section, 'rot_mat_row')
    config['rot_mat'] = (np.array([[rot_mat_row[0], rot_mat_row[1]],
                                   [-rot_mat_row[1], rot_mat_row[0]]])
                         / np.sqrt(rot_mat_row[0]**2 + rot_mat_row[1]**2))
    # lambda_min(z)
    surveyCutLambda_file = options.get_string(option_section, 'MCMF_lambda_min')
    tmp = np.genfromtxt(surveyCutLambda_file, names=True, dtype=None)
    config['survey_cut_richness'] = {}
    for name in tmp.dtype.names[1:]:
        config['survey_cut_richness'][name] = make_interp_spline(tmp['z'], tmp[name], k=1)
    # DES WL model
    DES_WL_priors_file = options.get_string(option_section, 'DES_WL_priors_file')
    with h5py.File(DES_WL_priors_file, 'r') as f:
        config['DES_WL_prior'] = {}
        for k in f.keys():
            config['DES_WL_prior'][k] = f[k][()]
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
              'DES_b_dev_1', 'DES_b_dev_2', 'DES_b_dev_m',
              'DES_s_dev', 'DES_s_dev_m',
              'DES_m_piv',
              'rhoSZrichness', 'rhoSZWL', 'rhoWLrichness',
              'z_DESWISE']:
        scaling[p] = block.get_double('mor_parameters', p)
    for p in config['DES_WL_prior'].keys():
        scaling['DES_%s' % p] = config['DES_WL_prior'][p]
    # Halo mass function
    z, M, N = block.get_grid('HMF', 'z_arr', 'M_arr', 'dNdlnM')
    HMF = {'z_arr': z, 'lnM_arr': np.log(M), 'dNdlnM': N}
    # Compute the expected ln mass
    M = stacked_mass.execute(HMF,
                             cosmology, scaling,
                             config['SPT_survey_tab'],
                             config['survey_cut_richness'], config['richness_scatter_model'],
                             config['z_bins'], config['rot_mat'], config['rot_bins_x'], config['rot_bins_y'])
    if np.any(np.isnan(M)):
        return 1
    block.put_double_array_1d('mean_mass', 'M', M.flatten())
    return 0


def cleanup(config):
    pass
