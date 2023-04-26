from __future__ import division
import numpy as np
from scipy.interpolate import interp1d
from cosmosis.datablock import option_section
import HMF_convo_SZrichness as HMF_convo

def setup(options):
    observable_pairs = options.get_string(option_section, 'observable_pairs').split()
    if len(observable_pairs)==1:
        pairs_zmin = [options.get_double(option_section, 'pairs_zmin')]
        pairs_zmax = [options.get_double(option_section, 'pairs_zmax')]
        pairs_Nz = [options.get_int(option_section, 'pairs_Nz')]
    else:
        pairs_zmin = options.get_double_array_1d(option_section, 'pairs_zmin')
        pairs_zmax = options.get_double_array_1d(option_section, 'pairs_zmax')
        pairs_Nz = options.get_int_array_1d(option_section, 'pairs_Nz')
        assert len(pairs_zmin)==len(observable_pairs), "Bad length of pairs_zmin"
        assert len(pairs_zmax)==len(observable_pairs), "Bad length of pairs_zmax"
        assert len(pairs_Nz)==len(observable_pairs), "Bad length of pairs_Nz"
    NPROC = options.get_int(option_section, 'NPROC', 0)

    surveyCutLambda_file = options.get_string(option_section, 'MCMF_lambda_min')
    tmp = np.loadtxt(surveyCutLambda_file, unpack=True)
    surveyCutLambda = {'shallow': interp1d(tmp[0], tmp[1], kind='linear'),
                       'deep': interp1d(tmp[0], tmp[2], kind='linear')}
    z_DESWISE = options.get_double(option_section, 'z_DESWISE')
    richness_scatter_model = options.get_string(option_section, 'richness_scatter_model')
    do_bias = options.get_bool(option_section, 'do_bias', False)

    multi_obs_convolution = HMF_convo.MultiObsConvolution(observable_pairs,
                                                          pairs_zmin, pairs_zmax, pairs_Nz,
                                                          surveyCutLambda, richness_scatter_model,
                                                          do_bias,
                                                          NPROC)

    return multi_obs_convolution

def execute(block, multi_obs_convolution):
    ##### Extract from datablock
    if multi_obs_convolution.do_bias:
        cosmology = {'Omega_l': block.get_double('cosmological_parameters', 'Omega_lambda'),
                     'h': block.get_double('cosmological_parameters', 'hubble')/100,
                     'w0': block.get_double('cosmological_parameters', 'w'),
                     'sigma8': block.get_double('cosmological_parameters', 'sigma_8')}
        for p in ['Omega_m', 'Omega_b', 'wa', 'n_s']:
            cosmology[p] = block.get_double('cosmological_parameters', p)
    else:
        cosmology = None
    scaling = {}
    for p in ['Bsz', 'Dsz',
              'Arichness', 'Brichness', 'Crichness', 'Drichness', 'richmPivot',
              'Arichness_ext', 'Brichness_ext', 'Crichness_ext', 'Drichness_ext',
              'rhoSZrichness',]:
        scaling[p] = block.get_double('mor_parameters', p)
    # Halo mass function
    z, M, N = block.get_grid('HMF', 'z_arr', 'M_arr', 'dNdlnM')
    HMF = {'z_arr': z, 'lnM_arr': np.log(M), 'dNdlnM': N}

    ##### Compute the convolutions
    dN_dmultiobs_dict = multi_obs_convolution.execute(HMF, scaling, cosmology)
    block.put_double_array_1d('dN_dmultiobs', 'lnM_arr', dN_dmultiobs_dict['lnM_arr'])
    for pair_name in multi_obs_convolution.observable_pairs:
        block.put_double_array_nd('dN_dmultiobs', pair_name, dN_dmultiobs_dict[pair_name])
        block.put_double_array_1d('dN_dmultiobs', '%s_z'%pair_name, dN_dmultiobs_dict['%s_z'%pair_name])

    return 0

def cleanup(config):
    pass
