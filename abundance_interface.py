from __future__ import division
import numpy as np
from astropy.table import Table

from cosmosis.datablock import option_section

import abundance as abundance_poisson
import abundance_covmat

def setup(options):
    ##### Global variables
    do_lambda_min = options.get_bool(option_section, 'lambda_min')
    do_covmat = options.get_bool(option_section, 'covmat')
    NPROC = options.get_int(option_section, 'NPROC')
    surveyCutSZmax = options.get_double(option_section, 'surveyCutSZmax')
    surveyCutRedshift = options.get_double_array_1d(option_section, 'surveyCutRedshift')
    # SPT survey
    SPT_survey_fields = options.get_string(option_section, 'SPT_survey_fields')
    SPT_survey = Table.read(SPT_survey_fields, format='ascii.commented_header')
    # Cluster catalog
    SPTcatalogfile = options.get_string(option_section, 'SPTcatalogfile')
    catalog = Table.read(SPTcatalogfile)
    ##### Initialize abundance
    if do_covmat:
        covmat_file = options.get_string(option_section, 'covmatfile')
        covmat = np.loadtxt(covmat_file)
        number_count = abundance_covmat.NumberCount(catalog, SPT_survey, covmat,
                                                    surveyCutSZmax, surveyCutRedshift,
                                                    NPROC)
    else:
        number_count = abundance_poisson.NumberCount(catalog, SPT_survey,
                                                     surveyCutSZmax, surveyCutRedshift,
                                                     NPROC)
    return number_count, do_lambda_min

def execute(block, stuff):
    number_count, do_lambda_min = stuff
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
    HMF = {'M_arr': block.get_double_array_1d('dN_dmultiobs', 'M_arr')}
    if do_lambda_min:
        z = {}
        for tmp in ['SZ_lambdacut_shallow', 'SZ_lambdacut_deep']:
            z['%s_z'%tmp] = block.get_double_array_1d('dN_dmultiobs', '%s_z'%tmp)
            HMF['%s_dNdlnM'%tmp] = block.get_double_array_nd('dN_dmultiobs', tmp)
        if np.all(z['SZ_lambdacut_shallow_z']==z['SZ_lambdacut_deep_z']):
            HMF['z_arr'] = z['SZ_lambdacut_shallow_z']
    else:
        HMF['z_arr'] = block.get_double_array_1d('dN_dmultiobs', 'SZ_z'),
        HMF['SZ_lambdacut_shallow_dNdlnM'] = block.get_double_array_nd('dN_dmultiobs', 'SZ')
        HMF['SZ_lambdacut_deep_dNdlnM'] = block.get_double_array_nd('dN_dmultiobs', 'SZ')
    HMF['len_z'] = len(HMF['z_arr'])
    # Compute the likelihood
    lnlike, dN_dz, dN_dz_500d, dN_dz_SZ, dN_dz_SPECS, dN_dxi, dN_dxi_500d, dN_dxi_SZ, dN_dxi_SPECS, N_total = number_count.lnlike(HMF, cosmology, scaling)
    if np.isneginf(lnlike):
        return 1
    for i,n in enumerate(dN_dz):
        block.put_double('dN', 'dN_dz_%d'%i, n)
    for i,n in enumerate(dN_dxi):
        block.put_double('dN', 'dN_dxi_%d'%i, n)
    block.put_double('likelihoods', 'ABUNDANCE_LIKE', lnlike)
    return 0

def cleanup(config):
    pass
