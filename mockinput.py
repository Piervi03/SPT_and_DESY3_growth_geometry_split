import h5py

SPT_survey = 'data/SPT_SZ_ECS_500d_survey.txt'
MCMF_lambda_min = 'data/MCMF_lambda_min.txt'

random_seed = 0

# Mira-Titan, Bocquet16, Tinker08
HMF = 'Tinker08'

# We mainly need the cosmology for E(z) and h
# (and for the DK15 c-M relation)
cosmology = {'Omega_m': .3, 'Ombh2': .022,
             'mnu': .06, 'nnu': 3.046,
             'h': 0.7,
             'w0': -1., 'wa': 0.,
             'n_s': .96, 'ln1e10As': 2.948}
cosmology['Omega_b'] = cosmology['Ombh2']/cosmology['h']**2
cosmology['Ommh2'] = cosmology['Omega_m']*cosmology['h']**2
cosmology['Omega_l'] = 1-cosmology['Omega_m']

scaling = {'Asz': .96, 'Bsz': 1.5, 'Csz': .5, 'Dsz': .2, 'zeta_min': 1.,
           'SPECS_calib': 1.05,
           'Esz': 0,
           'HSTbias': 0., 'HSTscatterLSS': 5.6e13,
           'MegacamBias': 0., 'MegacamScatterLSS': 6.3e13,
           'DWL_Megacam': .3, 'bWL_Megacam': 1,
           'DESbias': 0., 'DESscatterLSS': 6.3e13,

           'bWL_HST': 1., 'DWL_HST': .3,

           'DES_m_piv': 2e14,
           'DES_b_dev_1': 0., 'DES_b_dev_2': 0., 'DES_b_dev_m': 0.,
           'DES_s_dev': 0., 'DES_s_dev_m': 0.,
           'Euclid_m_piv': 2e14,
           'Euclid_b_dev': 0., 'Euclid_b_dev_m': 0.,
           'Euclid_s_dev': 0., 'Euclid_s_dev_m': 0.,

           'Adisp': 939., 'Bdisp': 2.91, 'Cdisp': .33, 'Ddisp0': .2, 'DdispN': 3.,

           'Arichness': 4.25, 'Brichness': 1., 'Crichness': 0., 'Drichness': .2,
           'Arichnessext': 4.25, 'Brichnessext': 1., 'Crichnessext': 0., 'Drichnessext': .2,
           'z_DESWISE': 1.1,

           'Ax': 6.5, 'Bx': .57, 'Cx': -.4, 'Dx': .12, 'Ex': 0,
           'slope_MgR': 1.16, 'slope_MgR_std': .016,
           'rhoSZrichness': 0., 'rhoSZdisp': 0., 'rhoSZX': 0., 'rhoSZWL': 0.,
           'rhoWLX': 0., 'rhoWLrichness': 0.,
           'rhoXrichness': 0,
           'SZmPivot': 3e14,
           'XraymPivot': 5e14,
           'richmPivot': 3e14,
           'YXPARAM': 'SPT_XVP',
           }
DES_WL_priors_file = './data/WLcalib_MCMF_dnf_500kpch.h5'
with h5py.File(DES_WL_priors_file, 'r') as f:
    for k in f.keys():
        scaling['DES_%s' % k] = f[k][()]
Euclid_WL_priors_file = './data/WLcalib_Euclid_baseline.h5'
with h5py.File(Euclid_WL_priors_file, 'r') as f:
    for k in f.keys():
        scaling['Euclid_%s' % k] = f[k][()]

# SPT survey cuts
z_cl_min_max = (0.25, 1.79)
# How to model X-ray profiles? 'PL' or 'beta'
profile_shape = 'PL'
# Observable errors
Xerr = .16
# Number of X-ray clusters
nXrayCluster = 80
# Use Mgas or Yx?
Xray_obs = 'Yx'
# Type of scatter in richness: 'lognormal', 'lognormalrelPoisson', 'lognormalGaussPoisson', 'lognormalGausssuperPoisson'
richness_scatter_model = 'lognormalrelPoisson'
