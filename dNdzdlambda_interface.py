import numpy as np
from scipy.interpolate import make_interp_spline
from astropy.table import Table

from cosmosis.datablock import option_section

import dNdzdlambda


def setup(options):
    # Global variables
    NPROC = options.get_int(option_section, 'NPROC')
    z_cl_min_max = options.get_double_array_1d(option_section, 'z_cl_min_max')
    # SPT survey
    SPT_survey_fields = options.get_string(option_section, 'SPT_survey_fields')
    SPT_survey = Table.read(SPT_survey_fields, format='ascii.commented_header')
    # Lambda cut
    lambda_min_file = options.get_string(option_section, 'MCMF_lambda_min')
    tmp = np.genfromtxt(lambda_min_file, names=True, dtype=None)
    lambda_min = {}
    for name in tmp.dtype.names[1:]:
        lambda_min[name] = make_interp_spline(tmp['z'], tmp[name], k=1)
    # Initialize abundance
    computer = dNdzdlambda.DistCompute(SPT_survey,
                                       z_cl_min_max,
                                       lambda_min,
                                       NPROC)
    return computer


def execute(block, computer):
    # Only need cosmo for E(z)-type stuff
    cosmology = {
                 'Omega_m': block.get_double('cosmological_parameters', 'Omega_m'),
                 'Omega_l': block.get_double('cosmological_parameters', 'omega_lambda'),
                 'w0': block.get_double('cosmological_parameters', 'w'),
                 'wa': block.get_double('cosmological_parameters', 'wa')}
    # SZ scaling relation parameters
    scaling = {}
    for p in ['Asz', 'Bsz', 'Csz', 'Dsz', 'Esz', 'SPECS_calib', 'SZmPivot', 'zeta_min',  'Delta_Csz_ECS', 'Delta_Csz_500d',
              'Arichness', 'Brichness', 'Crichness',
              'Arichness_ext', 'Brichness_ext', 'Crichness_ext',
              'richmPivot']:
        scaling[p] = block.get_double('mor_parameters', p)
    # Convolved halo mass function
    HMF = {'lnM_arr': block.get_double_array_1d('dN_dmultiobs', 'lnM_arr'),
           'z_arr': block.get_double_array_1d('dN_dmultiobs', 'z_arr'),
           'richness_SZ_lndNdlnM': block.get_double_array_nd('dN_dmultiobs', 'richness_SZ_lndNdlnM')}
    # Compute
    N_lambda = computer.run(HMF, cosmology, scaling)
    for i, n in enumerate(N_lambda):
        block.put_double('N', 'N_lambda_%d' % i, n)
    return 0


def cleanup(config):
    pass
