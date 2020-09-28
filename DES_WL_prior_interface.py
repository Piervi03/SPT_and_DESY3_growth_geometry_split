from __future__ import division
import numpy as np
import scipy.linalg as LA
import imp

from cosmosis.datablock import option_section

def setup(options):
    WLsimcalibfile = options.get_string(option_section, 'WLsimcalibfile')
    WLsimcalib = imp.load_source('WLsimcalib', WLsimcalibfile)

    tmp = np.loadtxt(WLsimcalib.WLcalibration['DES_WL_priors_file'])
    DES_WL_prior = {'mean': tmp[0,:],
                    'cov': tmp[1:,:]}

    return DES_WL_prior


def execute(block, DES_WL_prior):
    DES_var_names = ['DES_b_0', 'DES_s_0', 'DES_b_m',  'DES_s_m', 'DES_b_z', 'DES_s_z',]
    p = np.array([block.get_double('mor_parameters', p) for p in DES_var_names])

    resi = p - DES_WL_prior['mean']
    chi2 = np.dot(resi, LA.solve(DES_WL_prior['cov'], resi).T )
    lnlike = -.5*chi2

    block.put_double('likelihoods', 'DES_WL_PRIOR_LIKE', lnlike)

    return 0


def cleanup(config):
    pass
