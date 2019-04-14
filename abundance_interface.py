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
    scaling = {'SZmPivot': options.get_double(option_section, 'SZmPivot')}
    # SPT survey
    SPT_survey_fields = options.get_string(option_section, 'SPT_survey_fields')
    SPT_survey = Table.read(SPT_survey_fields, format='ascii.commented_header')
    # Cluster catalog
    SPTcatalogfile = options.get_string(option_section, 'SPTcatalogfile')
    catalog = Table.read(SPTcatalogfile)
    ##### Initialize abundance
    number_count = abundance.NumberCount(catalog, SPT_survey, scaling,
                                         surveyCutSZ, surveyCutRedshift,
                                         NPROC)

    return number_count

def execute(block, number_count):
    # Only need cosmo for E(z)-type stuff
    number_count.cosmology = {
        'Omega_m': block.get_double('cosmological_parameters', 'Omega_m'),
        'Omega_l': block.get_double('cosmological_parameters', 'omega_lambda'),
        'w0': block.get_double('cosmological_parameters', 'w'),
        'wa': block.get_double('cosmological_parameters', 'wa')}
    # SZ scaling relation parameters
    for p in ['Asz', 'Bsz', 'Csz', 'Dsz', 'Bsz2', 'Csz2', 'Esz', 'DszM', 'SPECS_calib']:
        number_count.scaling[p] = block.get_double('mor_parameters', p)
    # Halo mass function
    number_count.HMF = {
        'M_arr': block.get_double_array_1d('HMF', 'M_arr'),
        'z_arr': block.get_double_array_1d('HMF', 'z_arr'),
        'dNdlnM': block.get_double_array_nd('HMF', 'dNdlnM')}
    number_count.HMF['len_z'] = len(number_count.HMF['z_arr'])
    # Compute the likelihood
    lnlike = float(number_count.lnlike())
    if np.isneginf(lnlike):
        return 1
    block.put_double('likelihoods', 'ABUNDANCE_LIKE', lnlike)
    return 0

def cleanup(config):
    pass
