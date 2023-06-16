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
    z_DESWISE = options.get_double(option_section, 'z_DESWISE', default=surveyCutRedshift[1])
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
    return [number_count, do_lambda_min, z_DESWISE]


def execute(block, args):
    number_count, do_lambda_min, z_DESWISE = args
    # Only need cosmo for E(z)-type stuff
    cosmology = {
        'Omega_m': block.get_double('cosmological_parameters', 'Omega_m'),
        'Omega_l': block.get_double('cosmological_parameters', 'omega_lambda'),
        'w0': block.get_double('cosmological_parameters', 'w'),
        'wa': block.get_double('cosmological_parameters', 'wa')}
    # SZ scaling relation parameters
    scaling = {}
    for p in ['Asz', 'Bsz', 'Csz', 'Dsz', 'Esz', 'SPECS_calib', 'SZmPivot', 'zeta_min', 'Delta_Csz_ECS', 'Delta_Csz_500d']:
        scaling[p] = block.get_double('mor_parameters', p)
    # Convolved halo mass function
    HMF = {'lnM_arr': block.get_double_array_1d('dN_dmultiobs', 'lnM_arr'),
           'z_arr': block.get_double_array_1d('dN_dmultiobs', 'SZ_z'),
           'SZ_dNdlnM': block.get_double_array_nd('dN_dmultiobs', 'SZ')}
    if do_lambda_min:
        for depth in ['shallow', 'deep']:
            z, dNdlnM = {}, {}
            for opt_survey in ['base', 'ext']:
                if block.has_value('dN_dmultiobs', 'SZ_lambdacut_%s_%s_z'%(opt_survey, depth)):
                    z[opt_survey] = block.get_double_array_1d('dN_dmultiobs', 'SZ_lambdacut_%s_%s_z'%(opt_survey, depth))
                if block.has_value('dN_dmultiobs', 'SZ_lambdacut_%s_%s'%(opt_survey, depth)):
                    dNdlnM[opt_survey] = block.get_double_array_nd('dN_dmultiobs', 'SZ_lambdacut_%s_%s'%(opt_survey, depth))
            if 'base' in z.keys():
                if 'ext' in z.keys():
                    HMF['SZ_lambdacut_%s_z'%depth] = np.concatenate([z['base'][z['base']<z_DESWISE], z['ext'][z['ext']>=z_DESWISE]])
                    HMF['SZ_lambdacut_%s_dNdlnM'%depth] = np.concatenate([dNdlnM['base'][z['base']<z_DESWISE], dNdlnM['ext'][z['ext']>=z_DESWISE]])
                else:
                    HMF['SZ_lambdacut_%s_z'%depth] = z['base']
                    HMF['SZ_lambdacut_%s_dNdlnM'%depth] = dNdlnM['base']
            else:
                HMF['SZ_lambdacut_%s_z'%depth] = z['ext']
                HMF['SZ_lambdacut_%s_dNdlnM'%depth] = dNdlnM['ext']
            if not np.all(np.isclose(HMF['z_arr'], HMF['SZ_lambdacut_%s_z'%depth])):
                print("HMF z arrays do not match", depth)
                print(HMF['z_arr'])
                print(HMF['SZ_lambdacut_%s_z'%depth])
                return 1
    else:
        for tmp in ['SZ_lambdacut_shallow', 'SZ_lambdacut_deep']:
            HMF['%s_z'] = HMF['z_arr']
            HMF['%s_dNdlnM'%tmp] = HMF['SZ_dNdlnM']
    HMF['len_z'] = len(HMF['z_arr'])
    # Compute the likelihood
    lnlike, dN_dz, dN_dxi, N_total, all_lndNdxi = number_count.lnlike(HMF, cosmology, scaling)
    if np.isneginf(lnlike):
        return 1
    for i,n in enumerate(dN_dz):
        block.put_double('dN', 'dN_dz_%d'%i, n)
    for i,n in enumerate(dN_dxi):
        block.put_double('dN', 'dN_dxi_%d'%i, n)
    block.put_double_array_1d('cat', 'lndNdxi', all_lndNdxi)
    block.put_double('likelihoods', 'ABUNDANCE_LIKE', lnlike)
    return 0


def cleanup(config):
    pass
