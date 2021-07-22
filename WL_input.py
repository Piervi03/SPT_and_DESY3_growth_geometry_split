import numpy as np

random_seed = 0

Delta_crit = 200

DES = {'WL_z_max': .85,

      # DES Y3
      'source_p_arcmin2': 6,

      # From Grandis+19
      'shape_noise': .27,

      # Type of M-c scaling relation, 'Duffy08' or 'DK15' or 'Cihld18_obs' or float
      'mcType': 'Child18_obs',

      'boost': {'logc': .625, 'Blambda': .7774, 'corr_len': .0691, 'A_inf': -3,
               'A': [-2.12, -.355, 1.005, -.042, .877, -.455, 2.312, -2.995, 2.129, -2.959],
               'z_arr': np.linspace(.2, .9, 10)
               },

      'miscenter_opt': {
                       'kind': 'SPT',
                       'alpha_opt_0': 1.75, 'alpha_opt_lambda': 3.20, 'sigma_opt_0': -2.78, 'sigma_opt_1': -.43,
                       'kappa_SPT': 1.35, 'alpha_SZ_0': .47, 'SZ_comp0_0': -1.53, 'SZ_comp1_0': -.96
                       },

      'source_Pz_file': "2pt_NG_final_2ptunblind_11_13_20_wnz.fits",
      'source_weights_file': "tomo_weight_hist.txt",
      'tomo_bin_weight_file': "beta_boost_tomo_bin.npy",
      }

HST = {'shape_noise': .3,
       'source_p_arcmin2': 10.,
       'source_Pz_file': '/home/bocquet/SPT_cluster_data/HST_pz.txt',
       # Type of M-c scaling relation, 'Duffy08' or 'DK15' or 'Cihld18_obs' or float
       'mcType': 'DK15',
       }
