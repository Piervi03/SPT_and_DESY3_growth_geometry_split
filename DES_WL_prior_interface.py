from __future__ import division
import numpy as np
import scipy.linalg as LA

from cosmosis.datablock import option_section

def setup(options):
    # Posterior distribution file from Grandis&Bocquet et al. (2021)
    DES_WL_priors_file = options.get_string(option_section, 'DES_WL_priors_file')
    tmp = np.loadtxt(DES_WL_priors_file)
    DES_WL_prior = {'DESwl_z': [.252, .470, .783],
                    'mean': np.mean(tmp[:,:8], axis=0),
                    'cov': np.cov(tmp[:,:8], rowvar=False),}
    # Add systematic uncertainty due to hydro
    hydro_corr = {'b': .02, 'b_M': .018, 's': .25, 's_M': .59}
    DES_WL_prior['cov'][:3,:3]+= hydro_corr['b']**2 * np.eye(3)
    DES_WL_prior['cov'][3,3]+= hydro_corr['b_M']**2
    DES_WL_prior['cov'][4:7,4:7]+= hydro_corr['s']**2 * np.eye(3)
    DES_WL_prior['cov'][7,7]+= hydro_corr['s_M']**2
    
    # Individual components for ease of use
    DES_WL_prior['DESwl_bias_mean'] = DES_WL_prior['mean'][:3]
    DES_WL_prior['DESwl_scatter_mean'] = DES_WL_prior['mean'][4:7]
    DES_WL_prior['DESwl_bias_std'] = np.sqrt(np.diag(DES_WL_prior['cov'][:3,:3]))
    DES_WL_prior['DESwl_scatter_std'] = np.sqrt(np.diag(DES_WL_prior['cov'][4:7,4:7]))

    return DES_WL_prior


def execute(block, DES_WL_prior):
    # Create full array of entries as in posterior file
    dev = np.array([block.get_double('mor_parameters', 'DES_b_dev_%d'%i) for i in range(3)])
    b = DES_WL_prior['DESwl_bias_mean'] + dev*DES_WL_prior['DESwl_bias_std']
    dev = np.array([block.get_double('mor_parameters', 'DES_s_dev_%d'%i) for i in range(3)])
    s = DES_WL_prior['DESwl_scatter_mean'] + dev*DES_WL_prior['DESwl_scatter_std']
    p = [b[0], b[1], b[2], block.get_double('mor_parameters', 'DES_b_m'), s[0], s[1], s[2], block.get_double('mor_parameters', 'DES_s_m')]

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
