random_seed = 0

Delta_crit = 200

DES = {'WL_z_max': .94,
       # DES Y3
       'source_p_arcmin2': 6,
       # From Grandis+19
       'shape_noise': .375,
       # Type of M-c scaling relation, 'Duffy08' or 'DK15' or 'Cihld18_obs' or float
       'mcType': 3.5,
       # Boost and miscentering chains
       'DESboostfile': 'data/boost_smooth_dnf_MCMF.txt',
       'DESmiscenterfile': 'data/miscenter_SPTopt.txt',
       'DEScentertype': 'MCMF',
       # DES source photo-z
       'source_Pz_file': "data/2pt_NG_final_2ptunblind_11_13_20_wnz.fits",
       'source_weights_file': "data/tomo_weight_hist.txt",
       'Sigmacrit_file': "data/beta_tomo_bin.npy",
       }

Euclid = {'WL_z_max': 1.,
          'source_p_arcmin2': 20,
          # From Grandis+19
          'shape_noise': .375,
          # Type of M-c scaling relation, 'Duffy08' or 'DK15' or 'Cihld18_obs' or float
          'mcType': 3.5,
          # Boost and miscentering chains
          'DESboostfile': 'data/boost_smooth_dnf_MCMF.txt',
          'DESmiscenterfile': 'data/miscenter_SPTopt.txt',
          'DEScentertype': 'MCMF',
          'z_cl_offset': .1,
          }

HST = {'shape_noise': .3,
       'source_p_arcmin2': 10.,
       'source_Pz_file': 'data/HST_pz.txt',
       # Type of M-c scaling relation, 'Duffy08' or 'DK15' or 'Cihld18_obs' or float
       'mcType': 'DK15',
       }
