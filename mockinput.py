SPTdatafile = 'SPTcluster_data.py'

random_seed = 0

# We mainly need the cosmology for E(z) and h
# (and for the DK15 c-M relation)
cosmology = {'Omega_m': .3, 'Omega_l': .7, 'Omega_b': .04,
             'h': 0.7,
             'w0': -1., 'wa': 0.,
             'sigma8': .8, 'ns': .96,}

scaling = {'Asz': 4., 'Bsz': 1.34, 'Csz': .49, 'Dsz': .2,
           'Bsz2':0, 'Csz2':0, 'DszM':0, 'Esz':0,
           'WLbias': 0., 'WLscatter': 0.,
           'HSTbias': 0., 'HSTscatterLSS':5.6e13,
           'MegacamBias': 0., 'MegacamScatterLSS': 6.3e13,
           'DWL_Megacam': .3, 'bWL_Megacam': 1,
           'DESbias': 0., 'DESscatterLSS': 6.3e13,
           'Adisp':939., 'Bdisp':2.91, 'Cdisp':.33, 'Ddisp0':.2, 'DdispN':3.,
           'Arichness': 70., 'Brichness': 1., 'Crichness': 1., 'Drichness': .2,
           'Ax': 6.5, 'Bx': .57, 'Cx': -.4, 'Dx': .12, 'Ex':0,
           'slope_MgR': 1.16, 'slope_MgR_std': .016,
           'rhoSZrichness': 0., 'rhoSZdisp': 0., 'rhoSZX': 0., 'rhoSZWL': 0.,
           'rhoWLX': 0., 'rhoWLrichness':0.,
           'rhoXrichness': 0}

# SPT survey cuts
surveyCutSZ = (5., 47.)
surveyCutRedshift = (0.25, 2.)
# Pivot point of SZ scaling relation in solar masses
SZmPivot = 3e14
# Pivot point of X-ray scaling relation in solar masses
XraymPivot = 5e14
# Pivot point of richness scaling relation in solar masses
richmPivot = 3e14
# Type of M-c scaling relation, 'Duffy08' or 'DK15' or float
mcType = 'DK15'
# How to model X-ray profiles? 'PL' or 'beta'
profile_shape = 'PL'
# Observable errors
Xerr = .16
richness_err = 10
# Number of X-ray clusters
nXrayCluster = 80
# Use Mgas or Yx?
Xray_obs = 'Yx'
YXPARAM = 'SPT_XVP'
