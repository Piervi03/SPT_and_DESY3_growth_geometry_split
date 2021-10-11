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

      'boost': {
                'logc': 0.649133576266452,
                'Blambda': 0.6439579883982836,
                'corr_len': 0.0729655928917629,
                'A_inf': -3, 
                'A': [-3.3983177948846475, 0.9448875147198258, -0.5416372794150488, -0.20700121156068624, 0.1403094626423406, -0.7495373851259957, 1.000185766414055, -1.7301977606949952, 1.2623789006201902, -3.9512999404188003],
                'z_arr': np.linspace(.2, .9, 10)
               },

      'miscenter_opt': {
                        'kind': 'SPT',
                        'alpha_0': 0.7127904310586513, 'alpha_z': 0.27592357873626455, 'alpha_lam': 0.02277220223720074, 'comp0_0': 0.012661583694231636, 'comp0_z': -0.02857349851922065, 'comp0_lam': 0.27780185831905774, 'comp1_0': 0.07880481248876156, 'comp1_z': -0.16901046857333524, 'comp1_lam': -0.6562778664829945, 'kappa_SPT': 1.0128028775830318,
                        'B_lambda': 1, 'C_lambda': 0,
                       },

      'source_Pz_file': "2pt_NG_final_2ptunblind_11_13_20_wnz.fits",
      'source_weights_file': "tomo_weight_hist.txt",
      'tomo_bin_weight_file': "beta_tomo_bin.npy",
      }

HST = {'shape_noise': .3,
       'source_p_arcmin2': 10.,
       'source_Pz_file': '/home/bocquet/SPT_cluster_data/HST_pz.txt',
       # Type of M-c scaling relation, 'Duffy08' or 'DK15' or 'Cihld18_obs' or float
       'mcType': 'DK15',
       }
