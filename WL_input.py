import numpy as np

random_seed = 0

WL_z_max = 1

beta_bias_file = '../SPT_cluster_data/sci_uncertainty_zdiff_01_40bins_new.dat'

# Radial range, will be converted in degrees
r_minmax = .25, 3
r_edge_Mpc = np.linspace(r_minmax[0], r_minmax[1], 11)
# r_edge_Mpc = np.logspace(np.log10(r_minmax[0]),np.log10(r_minmax[1]),11)
r_bin_Mpc = np.sqrt(r_edge_Mpc[1:]**2 - r_edge_Mpc[:-1]**2)

# DES Y3
source_p_arcmin2 = 6

# Completely arbitrary
source_lognorm_dist_mean = .7
source_lognorm_dist_sigma = .5

# From Grandis+19
shape_noise = .27

# Offset in redshift behind the cluster
z_offset = .1
