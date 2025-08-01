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
    if SPTcatalogfile.endswith('.fits'):
        format_ = 'fits'
    elif SPTcatalogfile.endswith('.txt'):
        format_ = 'ascii'
    catalog = Table.read(SPTcatalogfile, format=format_)
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
    return [number_count, do_lambda_min]


def execute(block, args):
    number_count, do_lambda_min = args
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
    HMF = {'lnM_arr': block.get_double_array_1d('dN_dmultiobs', 'lnM_arr'),
           'z_arr': block.get_double_array_1d('dN_dmultiobs', 'z_arr'),
           'SZ_lndNdlnM': block.get_double_array_nd('dN_dmultiobs', 'SZ_lndNdlnM')}
    for name in ['SZ_lambdacut_shallow_lndNdlnM', 'SZ_lambdacut_deep_lndNdlnM']:
        if do_lambda_min:
            HMF[name] = block.get_double_array_nd('dN_dmultiobs', name)
        else:
            HMF[name] = HMF['SZ_lndNdlnM']
    # Compute the likelihood
    lnlike, N_z, N_xi, N_total, all_lndNdxi = number_count.lnlike(HMF, cosmology, scaling)
    if np.isneginf(lnlike):
        return 1
    for i,n in enumerate(N_z):
        block.put_double('N', 'N_z_%d'%i, n)
    for i,n in enumerate(N_xi):
        block.put_double('N', 'N_xi_%d'%i, n)
    block.put_double_array_1d('cat', 'lndNdxi', all_lndNdxi)
    block.put_double('likelihoods', 'ABUNDANCE_LIKE', lnlike)
    return 0


def cleanup(config):
    pass
