import numpy as np

random_seed = 0

Delta_crit = 200

DES = {'WL_z_max': .85,
       # DES Y3
       'source_p_arcmin2': 6,
       # From Grandis+19
       'shape_noise': .3,
       # Type of M-c scaling relation, 'Duffy08' or 'DK15' or 'Cihld18_obs' or float
       'mcType': 3.5,
       # Boost and miscentering chains
       'DESboostfile': '/archive1/users/bocquet/DESY3/WL_model/boost_smooth_dnf_SZ.txt',
       'DESmiscenterfile': '/archive1/users/bocquet/DESY3/WL_model/miscenter_3obs_mask_211020.txt',
       'DEScentertype': 'SPT',
       # DES source photo-z
       'source_Pz_file': "/archive1/users/bocquet/DESY3/2pt_NG_final_2ptunblind_11_13_20_wnz.fits",
       'source_weights_file': "/archive1/users/bocquet/DESY3/WL_model/tomo_weight_hist.txt",
       'tomo_bin_weight_file': "/archive1/users/bocquet/DESY3/WL_model/beta_tomo_bin.npy",
       }

HST = {'shape_noise': .3,
       'source_p_arcmin2': 10.,
       'source_Pz_file': '/home/bocquet/SPT_cluster_data/HST_pz.txt',
       # Type of M-c scaling relation, 'Duffy08' or 'DK15' or 'Cihld18_obs' or float
       'mcType': 'DK15',
       }

