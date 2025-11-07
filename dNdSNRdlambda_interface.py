import numpy as np
from scipy.interpolate import make_interp_spline
from astropy.table import Table

from cosmosis.datablock import option_section

import dNdSNRdlambda


def setup(options):
    config = {'NPROC': options.get_int(option_section, 'NPROC'),
              'richness_scatter_model': options.get_string(option_section, 'richness_scatter_model'),
              'z_cl_min_max': options.get_double_array_1d(option_section, 'z_cl_min_max'),
              'surveyCutSZmax': options.get_double(option_section, 'surveyCutSZmax'),
              'SPT_survey': Table.read(options.get_string(option_section, 'SPT_survey_fields'),
                                           format='ascii.commented_header')}
    tmp = options.get_double_array_1d(option_section, 'lambda_out')
    config['lambda_out'] = np.logspace(np.log10(tmp[0]), np.log10(tmp[1]), int(tmp[2]))
    tmp = options.get_double_array_1d(option_section, 'SNR_red_out')
    config['SNR_red_out'] = np.logspace(np.log10(tmp[0]), np.log10(tmp[1]), int(tmp[2]))
    # lambda_min(z)
    surveyCutLambda_file = options.get_string(option_section, 'MCMF_lambda_min')
    tmp = np.genfromtxt(surveyCutLambda_file, names=True, dtype=None)
    config['lambda_min'] = {}
    for name in tmp.dtype.names[1:]:
        config['lambda_min'][name] = make_interp_spline(tmp['z'], tmp[name], k=1)
    return config


def execute(block, config):
    # Only need cosmo for E(z)-type stuff
    cosmology = {'Omega_m': block.get_double('cosmological_parameters', 'Omega_m'),
                 'Omega_l': block.get_double('cosmological_parameters', 'omega_lambda'),
                 'w0': block.get_double('cosmological_parameters', 'w'),
                 'wa': block.get_double('cosmological_parameters', 'wa')}
    # Scaling relation parameters
    scaling = {}
    for p in ['Asz', 'Bsz', 'Csz', 'Dsz', 'Esz', 'SPECS_calib', 'SZmPivot', 'zeta_min',
              'Arichness', 'Brichness', 'Crichness', 'Drichness', 'richmPivot',
              'Arichness_ext', 'Brichness_ext', 'Crichness_ext', 'Drichness_ext',
              'z_DESWISE',]:
        scaling[p] = block.get_double('mor_parameters', p)
    # Convolved halo mass function
    HMF = {'lnM_arr': block.get_double_array_1d('dN_dmultiobs', 'lnM_arr'),
           'z_arr': block.get_double_array_1d('dN_dmultiobs', 'z_arr'),
           'richness_SZ_lndNdlnM': block.get_double_array_nd('dN_dmultiobs', 'richness_SZ_lndNdlnM')}
    # Compute
    res = dNdSNRdlambda.run(HMF, cosmology, scaling, **config)
    block.put_double_array_nd('dNdSdrichness', 'N', res)
    return 0


def cleanup(config):
    pass
