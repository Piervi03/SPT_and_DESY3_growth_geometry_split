import numpy as np
import h5py
from scipy.interpolate import make_interp_spline
from cosmosis.datablock import option_section
import HMF_convo


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

    if 'DESwl_richness_SZ_base' in observable_pairs:
        DES_WL_priors_file = options.get_string(option_section, 'DES_WL_priors_file')
        with h5py.File(DES_WL_priors_file, 'r') as f:
            DES_WL_prior = {}
            for k in f.keys():
                DES_WL_prior[k] = f[k][()]
    else:
        DES_WL_prior = None

    # lambda_min(z)
    surveyCutLambda_file = options.get_string(option_section, 'MCMF_lambda_min')
    tmp = np.genfromtxt(surveyCutLambda_file, names=True, dtype=None)
    surveyCutLambda = {}
    for name in tmp.dtype.names[1:]:
        surveyCutLambda[name] = make_interp_spline(tmp['z'], tmp[name], k=1)

    richness_scatter_model = options.get_string(option_section, 'richness_scatter_model')
    do_bias = options.get_bool(option_section, 'do_bias', False)

    multi_obs_convolution = HMF_convo.MultiObsConvolution(observable_pairs,
                                                          pairs_zmin, pairs_zmax, pairs_Nz,
                                                          surveyCutLambda, richness_scatter_model,
                                                          do_bias,
                                                          NPROC)

    return multi_obs_convolution, DES_WL_prior


def execute(block, stuff):
    multi_obs_convolution, DES_WL_prior = stuff
    # Extract from datablock
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
    for p in ['Bsz', 'Bx', 'Dsz',
              'Arichness', 'Brichness', 'Crichness', 'Drichness', 'richmPivot',
              'Arichness_ext', 'Brichness_ext', 'Crichness_ext', 'Drichness_ext',
              'rhoSZWL', 'rhoSZrichness', 'rhoWLrichness']:
        scaling[p] = block.get_double('mor_parameters', p)
    # DES lensing
    if DES_WL_prior is not None:
        for p in ['DES_b_dev_1', 'DES_b_dev_2', 'DES_b_dev_m',
                  'DES_s_dev', 'DES_s_dev_m',
                  'DES_m_piv']:
            scaling[p] = block.get_double('mor_parameters', p)
        for p in DES_WL_prior.keys():
            scaling['DES_%s'%p] = DES_WL_prior[p]
    # Halo mass function
    z, M, N = block.get_grid('HMF', 'z_arr', 'M_arr', 'dNdlnM')
    HMF = {'z_arr': z, 'lnM_arr': np.log(M), 'dNdlnM': N}

    # Compute the convolutions
    dN_dmultiobs_dict = multi_obs_convolution.execute(HMF, scaling, cosmology)
    block.put_double_array_1d('dN_dmultiobs', 'lnM_arr', dN_dmultiobs_dict['lnM_arr'])
    for pair_name in multi_obs_convolution.observable_pairs:
        block.put_double_array_nd('dN_dmultiobs', pair_name, dN_dmultiobs_dict[pair_name])
        block.put_double_array_1d('dN_dmultiobs', '%s_z'%pair_name, dN_dmultiobs_dict['%s_z'%pair_name])

    return 0


def cleanup(config):
    pass
