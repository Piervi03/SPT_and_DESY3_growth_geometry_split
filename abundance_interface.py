from __future__ import division
import numpy as np
from astropy.table import Table
import os

from cosmosis.datablock import option_section

import abundance

def setup(options):
    number_count = abundance.NumberCount(options)
    ##### Global variables
    number_count.NPROC = options.get_int(option_section, 'NPROC')
    number_count.surveyCutSZ = options.get_double_array_1d(option_section, 'surveyCutSZ')
    number_count.surveyCutRedshift = options.get_double_array_1d(option_section, 'surveyCutRedshift')
    number_count.scaling = {'SZmPivot': options.get_double(option_section, 'SZmPivot')}
    # SPT survey
    SPT_survey_fields = options.get_string(option_section, 'SPT_survey_fields')
    assert os.path.isfile(SPT_survey_fields), "SPT survey table does not exist"
    number_count.SPT_survey = Table.read(SPT_survey_fields, format='ascii.commented_header')
    # Cluster catalog
    SPTcatalogfile = options.get_string(option_section, 'SPTcatalogfile')
    assert os.path.isfile(SPTcatalogfile), "SPT catalog file does not exist"
    number_count.catalog = Table.read(SPTcatalogfile)
    ##### Various observable arrays
    # Lin spaced for convo with unit scatter (+3 sigma margin)
    Nxi = int((number_count.surveyCutSZ[1]+3 - 2.7)/.1 + 1)
    number_count.xi_bins = np.linspace(2.7, number_count.surveyCutSZ[1]+3, Nxi)
    number_count.dxi = number_count.xi_bins[1] - number_count.xi_bins[0]
    # ln(zeta(xi_bins))
    number_count.ln_zeta_xi_arr = np.log(scaling_relations.xi2zeta(number_count.xi_bins))
    # dlnzeta/dxi (xi_bins)
    number_count.dlnzeta_dxi_arr = scaling_relations.dlnzeta_dxi(number_count.xi_bins)
    # Arrays over which we'll integrate (survey cuts applied)
    Nxi = int(np.log10(number_count.surveyCutSZ[1]/number_count.surveyCutSZ[0])/.005 + 1)
    number_count.xi_arr = np.logspace(np.log10(number_count.surveyCutSZ[0]), np.log10(number_count.surveyCutSZ[1]), Nxi)
    dz = .01
    Nz = int((number_count.surveyCutRedshift[1]-number_count.surveyCutRedshift[0])/dz + 1)
    number_count.z_arr = np.linspace(number_count.surveyCutRedshift[0], number_count.surveyCutRedshift[1], Nz)

    return number_count

def execute(block, number_count):
    # Only need cosmo for E(z)-type stuff
    number_count.cosmology = {
        'Omega_m': block.get_double('cosmological_parameters', 'Omega_m'),
        'Omega_l': block.get_double('cosmological_parameters', 'omega_lambda'),
        'w0': block.get_double('cosmological_parameters', 'w'),
        'wa': block.get_double('cosmological_parameters', 'wa')}
    # SZ scaling relation parameters
    for p in ['Asz', 'Bsz', 'Csz', 'Dsz', 'Bsz2', 'Csz2', 'Esz', 'DszM']:
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
