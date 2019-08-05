import numpy as np

random_seed = 0

Delta_crit = 500

WL_z_max = 1

beta_bias_file = '../SPT_cluster_data/sci_uncertainty_zdiff_01_40bins_new.dat'

# Radial range, will be converted in arcmin
r_minmax = .25, 3
r_edge_Mpc = np.linspace(r_minmax[0], r_minmax[1], 11)
r_bin_Mpc = np.array([np.sum(r_edge_Mpc[i:i+2]**3)/np.sum(r_edge_Mpc[i:i+2]**2)
                      for i in range(len(r_edge_Mpc)-1)])

# Miscentering parameters [arcmin]
rho0 = .6
sigma0 = .1
sigma1 = .2

# DES Y3
source_p_arcmin2 = 30

# Completely arbitrary
source_lognorm_dist_mean = .7
source_lognorm_dist_sigma = .5

# From Grandis+19
shape_noise = .27

# Offset in redshift behind the cluster
z_offset = .1
