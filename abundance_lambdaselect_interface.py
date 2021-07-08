from __future__ import division
import numpy as np
from astropy.table import Table

from cosmosis.datablock import option_section

import abundance

def setup(options):
    ##### Global variables
    NPROC = options.get_int(option_section, 'NPROC')
    surveyCutSZ = options.get_double_array_1d(option_section, 'surveyCutSZ')
    surveyCutRedshift = options.get_double_array_1d(option_section, 'surveyCutRedshift')
    # SPT survey
    SPT_survey_fields = options.get_string(option_section, 'SPT_survey_fields')
    SPT_survey = Table.read(SPT_survey_fields, format='ascii.commented_header')
    # Cluster catalog
    SPTcatalogfile = options.get_string(option_section, 'SPTcatalogfile')
    catalog = Table.read(SPTcatalogfile)
    ##### Initialize abundance
    number_count = abundance.NumberCount(catalog, SPT_survey,
                                         surveyCutSZ, surveyCutRedshift,
                                         NPROC)

    return number_count

def execute(block, number_count):
    # Only need cosmo for E(z)-type stuff
    cosmology = {
        'Omega_m': block.get_double('cosmological_parameters', 'Omega_m'),
        'Omega_l': block.get_double('cosmological_parameters', 'omega_lambda'),
        'w0': block.get_double('cosmological_parameters', 'w'),
        'wa': block.get_double('cosmological_parameters', 'wa')}
    # SZ scaling relation parameters
    scaling = {}
    for p in ['Asz', 'Bsz', 'Csz', 'Dsz', 'Esz', 'SPECS_calib', 'SZmPivot', 'zeta_min']:
        scaling[p] = block.get_double('mor_parameters', p)
    # Convolved halo mass function
    HMF = {'M_arr': block.get_double_array_1d('dN_dmultiobs', 'M_arr'),}
    z = {}
    for tmp in ['SZ_lambdacut_shallow', 'SZ_lambdacut_deep']:
        z['%s_z'%tmp] = block.get_double_array_1d('dN_dmultiobs', '%s_z'%tmp)
        HMF['%s_dNdlnM'%tmp] = block.get_double_array_nd('dN_dmultiobs', tmp)
    if np.all(z['SZ_lambdacut_shallow_z']==z['SZ_lambdacut_deep_z']):
        HMF['z_arr'] = z['SZ_lambdacut_shallow_z']
    HMF['len_z'] = len(HMF['z_arr'])
    # Compute the likelihood
    lnlike = float(number_count.lnlike(HMF, cosmology, scaling))
    if np.isneginf(lnlike):
        return 1
    block.put_double('likelihoods', 'ABUNDANCE_LIKE', lnlike)
    return 0

def cleanup(config):
    pass
