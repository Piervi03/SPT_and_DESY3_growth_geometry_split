import numpy as np

random_seed = 0

Delta_crit = 200

WL_z_max = .9

beta_bias_file = '/home/bocquet/SPT_cluster_data/sci_uncertainty_zdiff_01_40bins_new.dat'

# LSS covariance matrix
LSS_cov_z_file = '/home/bocquet/clustercosmo19/mock_DESWL/z_cluster.npy'
LSS_cov_file = '/home/bocquet/clustercosmo19/mock_DESWL/LSS_covs.npy'

# DES Y3
source_p_arcmin2 = 10

# Completely arbitrary
source_lognorm_dist_mean = .7
source_lognorm_dist_sigma = .5

# From Grandis+19
shape_noise = .27

# Offset in redshift behind the cluster
z_offset = .1

WL_params = {'x0_fcl': 1.01,
             'lambda_piv_fcl': 67.,
             'A_fcl': [.087, .12, .161, .105, .056, .079, .108, .089],
             'A_fcl_z': [.15, .2, .25, .3, .4, .5, .6, .7, .9],
             'B_fcl': .515,
             'c_fcl': 1.91,
             'DES_WL_priors_file': '/home/bocquet/SPT_cluster_data/DES_Y1_MariaPaulus/WL_priors.txt',
             'miscenter_opt': {'rho': 0.06910239, 'sigma0': 0.31633618, 'sigma0_z': 0.5141243, 'sigma0_lam': -0.61753193, 'sigma1': 0.03232347, 'sigma1_lam': -1.60232199,
                               'A_lambda': 60, 'B_lambda': 1., 'C_lambda': 0},
                              }
