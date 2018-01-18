SPTdatafile = 'SPTcluster_data.py'

random_seed = 0

cosmology = {'Nnumass': 0., 'Nnurel': 3.046, 'Mnu': 0.0,
             'Omega_m': .3, 'Omega_l': .7, 'Omega_b': .04,
             'h': 0.7,
             'ns': .96,
             'w0': -1., 'wa': 0.,
             'sigma8': .8159,
             'tau': .089,
             'gamma': 0.55,
             'lnAs': 2.919,
             'Pnorm0': 1., 'Pnorm1': 1., 'Pnorm2': 1., 'Pnorm3': 1.,
             'Omch2': .1, 'Ombh2': .02202,
             'Omega_nu': 0.,
             'theta': 1.03}

scaling = {'Asz': 4., 'Bsz': 1.34, 'Csz': .49, 'Dsz': .2,
           'Bsz2':0, 'Csz2':0, 'DszM':0, 'Esz':0,
           'WLbias': 0., 'WLscatter': 0.,
           'HSTbias': 0., 'HSTscatterLSS':5.6e13,
           'MegacamBias': 0., 'MegacamScatterLSS': 6.3e13,
           'DWL_Megacam': .3, 'bWL_Megacam': 1,
           'DESbias': 0., 'DESscatterLSS': 6.3e13,
           'Adisp':939., 'Bdisp':2.91, 'Cdisp':.33, 'Ddisp0':.2, 'DdispN':3.,
           'Alambda': 70., 'Blambda': 1., 'Clambda': 1., 'Dlambda': .2,
           'rhoSZlambda': 0., 'rhoSZdisp': 0., 'rhoSZX': 0.,
           'rhoSZWL': 0., 'rhoWLX': 0., 'rhoXdisp': 0.,
           'Ax': 6.5, 'Bx': .57, 'Cx': -.4, 'Dx': .12, 'Ex':0,
           'slope_MgR':1.16}

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
# X-ray obs error
Xerr = .16
# Number of X-ray clusters
nXrayCluster = 80
