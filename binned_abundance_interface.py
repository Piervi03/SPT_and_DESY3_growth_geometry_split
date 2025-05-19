import numpy as np
from astropy.table import Table

from cosmosis.datablock import option_section

import binned_abundance


def setup(options):
    SPT_survey_fields = options.get_string(option_section, 'SPT_survey_fields')
    kwargs = {'do_lambda_min': options.get_bool(option_section, 'lambda_min'),
              'NPROC': options.get_int(option_section, 'NPROC'),
              'z_DESWISE':  options.get_double(option_section, 'z_DESWISE'),
              'SPT_survey_tab': Table.read(SPT_survey_fields, format='ascii.commented_header'),
              'z_bins': options.get_double_array_1d(option_section, 'z_bins'),
              'SNR_bins': options.get_double_array_1d(option_section, 'SNR_bins')}
    return kwargs


def execute(block, kwargs):
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
    if kwargs['do_lambda_min']:
        for depth in ['shallow', 'deep']:
            z, dNdlnM = {}, {}
            for opt_survey in ['base', 'ext']:
                if block.has_value('dN_dmultiobs', 'SZ_lambdacut_%s_%s_z' % (opt_survey, depth)):
                    z[opt_survey] = block.get_double_array_1d('dN_dmultiobs', 'SZ_lambdacut_%s_%s_z' % (opt_survey, depth))
                if block.has_value('dN_dmultiobs', 'SZ_lambdacut_%s_%s' % (opt_survey, depth)):
                    dNdlnM[opt_survey] = block.get_double_array_nd('dN_dmultiobs', 'SZ_lambdacut_%s_%s' % (opt_survey, depth))
            if 'base' in z.keys():
                if 'ext' in z.keys():
                    HMF['SZ_lambdacut_%s_z' % depth] = np.concatenate([z['base'][z['base'] < kwargs['z_DESWISE']],
                                                                       z['ext'][z['ext'] >= kwargs['z_DESWISE']]])
                    HMF['SZ_lambdacut_%s_dNdlnM' % depth] = np.concatenate([dNdlnM['base'][z['base'] < kwargs['z_DESWISE']],
                                                                            dNdlnM['ext'][z['ext'] >= kwargs['z_DESWISE']]])
                else:
                    HMF['SZ_lambdacut_%s_z' % depth] = z['base']
                    HMF['SZ_lambdacut_%s_dNdlnM' % depth] = dNdlnM['base']
            else:
                HMF['SZ_lambdacut_%s_z' % depth] = z['ext']
                HMF['SZ_lambdacut_%s_dNdlnM' % depth] = dNdlnM['ext']
            if not np.all(np.isclose(HMF['z_arr'], HMF['SZ_lambdacut_%s_z' % depth])):
                print("HMF z arrays do not match", depth)
                print(HMF['z_arr'])
                print(HMF['SZ_lambdacut_%s_z' % depth])
                return 1
    else:
        for tmp in ['SZ_lambdacut_shallow', 'SZ_lambdacut_deep']:
            HMF['%s_z'] = HMF['z_arr']
            HMF['%s_dNdlnM' % tmp] = HMF['SZ_dNdlnM']
    # Compute the expected number counts
    N = binned_abundance.execute(HMF, cosmology, scaling,
                                 kwargs['SPT_survey_tab'],
                                 kwargs['z_bins'], kwargs['SNR_bins'],
                                 kwargs['NPROC'])
    if np.any(np.isnan(N)):
        return 1
    block.put_double_array_1d('SPT_cluster', 'N', N)
    return 0


def cleanup(config):
    pass
