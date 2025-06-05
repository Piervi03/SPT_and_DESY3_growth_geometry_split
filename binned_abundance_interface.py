import numpy as np

from cosmosis.datablock import option_section

import binned_abundance


def setup(options):
    SPT_survey_fields = options.get_string(option_section, 'SPT_survey_fields')
    config = {'do_lambda_min': options.get_bool(option_section, 'lambda_min'),
              'NPROC': options.get_int(option_section, 'NPROC'),
              'SPT_survey_tab': np.genfromtxt(SPT_survey_fields, names=True, dtype=None),
              'z_bins': options.get_double_array_1d(option_section, 'SPTcl_z_bins'),
              'SNR_bins': options.get_double_array_1d(option_section, 'SPTcl_SNR_bins')}
    return config


def execute(block, config):
    # Only need cosmo for E(z)-type stuff
    cosmology = {'Omega_m': block.get_double('cosmological_parameters', 'Omega_m'),
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
        if config['do_lambda_min']:
            HMF[name] = block.get_double_array_nd('dN_dmultiobs', name)
        else:
            HMF[name] = HMF['SZ_lndNdlnM']
    # Compute the expected number counts
    N = binned_abundance.execute(HMF, cosmology, scaling,
                                 config['SPT_survey_tab'],
                                 config['z_bins'], config['SNR_bins'],
                                 config['NPROC'])
    if np.any(np.isnan(N)):
        return 1
    block.put_double_array_1d('SPT_cluster', 'N', N)
    return 0


def cleanup(config):
    pass
