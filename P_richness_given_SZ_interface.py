import numpy as np
from scipy.interpolate import make_interp_spline
from astropy.table import Table

from cosmosis.datablock import option_section

import P_richness_given_SZ


def setup(options):
    catalog = Table.read(options.get_string(option_section, 'SPTcatalogfile'))
    catalog = catalog[catalog['COSMO_SAMPLE'] == 1]
    SPT_survey_fields = options.get_string(option_section, 'SPT_survey_fields')
    surveyCutLambda_file = options.get_string(option_section, 'MCMF_lambda_min')
    tmp = np.loadtxt(surveyCutLambda_file, unpack=True)
    surveyCutRichness = {'shallow': make_interp_spline(tmp[0], tmp[1], k=1),
                         'deep': make_interp_spline(tmp[0], tmp[2], k=1)}
    config = {'catalog': catalog,
              'NPROC': options.get_int(option_section, 'NPROC'),
              'surveyCutRichness': surveyCutRichness,
              'SPT_survey_tab': np.genfromtxt(SPT_survey_fields, names=True, dtype=None)}
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
                                        config['surveyCutRichness'],
                                        config['NPROC'])
    # Finalize
    if not np.isfinite(lnlike):
        return 1
    block.put_double('likelihoods', 'P_richness_given_xi_like', lnlike)
    return 0


def cleanup(config):
    pass
