from __future__ import division
import numpy as np
import scipy.linalg as LA

from cosmosis.datablock import option_section

def setup(options):
    # Posterior distribution file from Grandis&Bocquet et al. (2021)
    DES_WL_priors_file = options.get_string(option_section, 'DES_WL_priors_file')

    tmp = np.loadtxt(DES_WL_priors_file)
    DES_WL_prior = {'mean': np.mean(tmp[:,:10], axis=0),
                    'cov': np.cov(tmp[:,:10], rowvar=False),
                    'DESwl_z': [.252, .470, .783, 1.18],
                    'DESwl_bias_mean': np.mean(tmp[:,:4], axis=0),
                    'DESwl_bias_std': np.std(tmp[:,:4], axis=0),
                    'DESwl_scatter_mean': np.mean(tmp[:,5:9], axis=0),
                    'DESwl_scatter_std': np.std(tmp[:,5:9], axis=0)}

    return DES_WL_prior


def execute(block, DES_WL_prior):
    # Create full array of entries as in posterior file
    b = DES_WL_prior['DESwl_bias_mean'] + block.get_double('mor_parameters', 'DES_b_dev')*DES_WL_prior['DESwl_bias_std']
    s = DES_WL_prior['DESwl_scatter_mean'] + block.get_double('mor_parameters', 'DES_s_dev')*DES_WL_prior['DESwl_scatter_std']
    p = [b[0], b[1], b[2], b[3], block.get_double('mor_parameters', 'DES_b_m'), s[0], s[1], s[2], s[3], block.get_double('mor_parameters', 'DES_s_m')]

    # Likelihood
    resi = np.array(p) - DES_WL_prior['mean']
    chi2 = np.dot(resi, LA.solve(DES_WL_prior['cov'], resi).T )
    lnlike = -.5*chi2
    block.put_double('likelihoods', 'DES_WL_PRIOR_LIKE', lnlike)

    # Write effective WL scaling relation to block. Not very elegant but better than hard-coding.
    for p in ['DESwl_z', 'DESwl_bias_mean', 'DESwl_bias_std', 'DESwl_scatter_mean', 'DESwl_scatter_std']:
        block.put_double_array_1d('mor_parameters', p, DES_WL_prior[p])

    return 0


def cleanup(config):
    pass
