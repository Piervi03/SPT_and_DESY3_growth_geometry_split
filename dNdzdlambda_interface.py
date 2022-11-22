from __future__ import division
import numpy as np
from scipy.interpolate import interp1d
from astropy.table import Table

from cosmosis.datablock import option_section

import dNdzdlambda

def setup(options):
    ##### Global variables
    NPROC = options.get_int(option_section, 'NPROC')
    surveyCutRedshift = options.get_double_array_1d(option_section, 'surveyCutRedshift')
    # SPT survey
    SPT_survey_fields = options.get_string(option_section, 'SPT_survey_fields')
    SPT_survey = Table.read(SPT_survey_fields, format='ascii.commented_header')
    # Lambda cut
    surveyCutLambda_file = options.get_string(option_section, 'MCMF_lambda_min')
    tmp = np.loadtxt(surveyCutLambda_file, unpack=True)
    surveyCutLambda = {'shallow': interp1d(tmp[0], tmp[1], kind='linear'),
                       'deep': interp1d(tmp[0], tmp[2], kind='linear')}
    ##### Initialize abundance
    computer = dNdzdlambda.DistCompute(SPT_survey,
                                       surveyCutRedshift,
                                       surveyCutLambda,
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
    for p in ['Asz', 'Bsz', 'Csz', 'Dsz', 'Esz', 'SPECS_calib', 'SZmPivot', 'zeta_min',
              'Arichness', 'Brichness', 'Crichness', 'richmPivot']:
        scaling[p] = block.get_double('mor_parameters', p)
    # Convolved halo mass function
    HMF = {'M_arr': block.get_double_array_1d('dN_dmultiobs', 'M_arr'),
           'z_arr': block.get_double_array_1d('dN_dmultiobs', 'richness_SZ_z'),
           'dNdlnM': block.get_double_array_nd('dN_dmultiobs', 'richness_SZ')}
    HMF['len_z'] = len(HMF['z_arr'])
    # Compute
    dN_dz, dN_dlambda = computer.run(HMF, cosmology, scaling)
    #for i,n in enumerate(dN_dz):
    #    block.put_double('dN', 'dN_dz_%d'%i, n)
    for i,n in enumerate(dN_dlambda):
        block.put_double('dN', 'dN_dlambda_%d'%i, n)
    return 0

def cleanup(config):
    pass
