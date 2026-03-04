import numpy as np
from scipy.interpolate import make_interp_spline
from astropy.table import Table

from cosmosis.datablock import option_section

import P_richness_given_SZ


def setup(options):
    config = {'NPROC': options.get_int(option_section, 'NPROC'),
              'richness_scatter_model': options.get_string(option_section, 'richness_scatter_model'),
              'SPT_survey_tab': Table.read(options.get_string(option_section, 'SPT_survey_fields'),
                                           format='ascii.commented_header')}
    # Catalog
    z_cl_min_max = options.get_double_array_1d(option_section, 'z_cl_min_max')
    catalog = Table.read(options.get_string(option_section, 'SPTcatalogfile'))
    xi_cut = [xi >= config['SPT_survey_tab']['XI_MIN'][config['SPT_survey_tab']['FIELD'] == field][0]
              for xi, field in zip(catalog['XI'], catalog['FIELD'])]
    config['catalog'] = catalog[(catalog['COSMO_SAMPLE'] == 1)
                                & xi_cut
                                & (catalog['MASK_FRACTION_60'] < options.get_double(option_section, 'MASK_FRACTION_60', 1.))
                                & (z_cl_min_max[0] <= catalog['REDSHIFT'])
                                & (catalog['REDSHIFT'] <= z_cl_min_max[1])]
    # lambda_min(z)
    surveyCutLambda_file = options.get_string(option_section, 'MCMF_lambda_min')
    tmp = np.genfromtxt(surveyCutLambda_file, names=True, dtype=None)
    config['lambda_min'] = {}
    for name in tmp.dtype.names[1:]:
        config['lambda_min'][name] = make_interp_spline(tmp['z'], tmp[name], k=1)
    return config


def execute(block, config):
    # Only need cosmo for E(z)-type stuff
    cosmology = {'Omega_m': block.get_double('cosmological_parameters', 'Omega_m'),
                 'Omega_l': block.get_double('cosmological_parameters', 'omega_lambda'),
                 'w0': block.get_double('cosmological_parameters', 'w'),
                 'wa': block.get_double('cosmological_parameters', 'wa')}
    # Scaling relation parameters
    scaling = {}
    for p in ['Asz', 'Bsz', 'Csz', 'Dsz', 'Esz', 'SPECS_calib', 'SZmPivot', 'zeta_min',
              'Arichness', 'Brichness', 'Crichness', 'Drichness', 'richmPivot',
              'Arichness_ext', 'Brichness_ext', 'Crichness_ext', 'Drichness_ext',
              'z_DESWISE',]:
        scaling[p] = block.get_double('mor_parameters', p)
    # Convolved halo mass function
    HMF = {'lnM_arr': block.get_double_array_1d('dN_dmultiobs', 'lnM_arr'),
           'z_arr': block.get_double_array_1d('dN_dmultiobs', 'z_arr'),
           'richness_SZ_lndNdlnM': block.get_double_array_nd('dN_dmultiobs', 'richness_SZ_lndNdlnM')}
    # Compute the likelihood
    lnlike = P_richness_given_SZ.lnlike(config['catalog'], config['SPT_survey_tab'],
                                        HMF,
                                        cosmology, scaling,
                                        config['lambda_min'], config['richness_scatter_model'],
                                        config['NPROC'])
    # Finalize
    if not np.isfinite(lnlike):
        return 1
    block.put_double('likelihoods', 'P_richness_given_xi_like', lnlike)
    return 0


def cleanup(config):
    pass
