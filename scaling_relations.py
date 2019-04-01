from __future__ import division
import numpy as np

import cosmo

####################
##### xi--zeta relations
def xi2zeta(xi):
    return (xi**2 - 3)**.5
def zeta2xi(zeta):
    return (zeta**2 + 3)**.5
def dlnzeta_dxi(xi):
    return xi / (xi**2 - 3)


####################
def mass2obs(name, mass, z, scaling, cosmology):
    """Returns observable given (mass, z) using scaling relation."""
    if name=='zeta':
        lnzeta = np.log(scaling['Asz']) + scaling['Bsz'] * np.log(mass/scaling['SZmPivot'])\
            + scaling['Csz'] * np.log(cosmo.Ez(z, cosmology)/cosmo.Ez(.6, cosmology))\
            + scaling['Esz'] * np.log(mass/scaling['SZmPivot'])*np.log(cosmo.Ez(z, cosmology)/cosmo.Ez(.6, cosmology))
        return np.exp(lnzeta)
    elif name=='Yx':
        if scaling['YXPARAM']=='SPT_XVP':
            return 3 * (cosmology['h']/.7)**-2.5 * (mass/1e14 /.7**(3/2) / scaling['Ax'] \
                / cosmo.Ez(z, cosmology)**scaling['Cx'])**(1/scaling['Bx'])
        elif scaling['YXPARAM']=='obs-mass':
            return scaling['Ax'] * (cosmology['h']/.7)**-2.5 * (mass/cosmology['h']/scaling['XraymPivot'])**scaling['Bx'] \
                * (cosmo.Ez(z, cosmology)/cosmo.Ez(.6, cosmology))**scaling['Cx']
    elif name=='Mgas':
        lnMgas = np.log(scaling['XraymPivot'] * scaling['Ax']) - 2.5 * np.log(cosmology['h']/.7) \
            + scaling['Bx'] * np.log(mass/scaling['XraymPivot']/cosmology['h'])\
            + scaling['Cx'] * np.log(cosmo.Ez(z, cosmology)/cosmo.Ez(.6, cosmology))\
            + scaling['Ex'] * np.log(mass/scaling['XraymPivot']/cosmology['h'])*np.log(cosmo.Ez(z, cosmology)/cosmo.Ez(.6, cosmology))
        return np.exp(lnMgas)
    elif name=='disp':
        h70z = cosmology['h']/.7*cosmo.Ez(z, cosmology)
        M200c = np.exp(lnM500_to_lnM200(z, np.log(mass)))
        if len(M200c)==1: M200c = M200c[0]
        return scaling['Adisp'] * (M200c/1e15/cosmology['h'])**(1/scaling['Bdisp']) * h70z**scaling['Cdisp']
    elif name=='richness':
        return scaling['Arichness'] * (mass/scaling['richmPivot'])**scaling['Brichness']\
            * (cosmo.Ez(z, cosmology)/cosmo.Ez(.6, cosmology))**scaling['Crichness']
    elif name=='WLMegacam':
        return scaling['bWL_Megacam'] * mass
    elif name=='WLHST':
        return scaling['bWL_HST'] * mass
    elif name=='WLDES':
        return scaling['bWL_DES'] * mass
    else:
        raise ValueError("Observable not known:",name)


####################
def dlnM_dlnobs(name, scaling, cosmology=None, M0_arr=None, z=None):
    """Returns dlnM/dln(obs) for a given observable."""
    if name=='zeta':
        return 1/scaling['Bsz']
    elif name=='richness':
        return 1/scaling['Brichness']
    elif name=='Yx':
        if scaling['YXPARAM']=='SPT_XVP':
            return 1/(1/scaling['Bx'] - scaling['dlnMg_dlnr']/3)
        elif scaling['YXPARAM']=='obs-mass':
            return 1/(scaling['Bx'] - scaling['dlnMg_dlnr']/3)
    elif name=='Mgas':
        return 1/(scaling['Bx'] - scaling['dlnMg_dlnr']/3)
    elif (name=='WLMegacam')|(name=='WLHST')|(name=='WLDES'):
        return 1.
    elif name=='disp':
        dlnM = np.log(1.01)
        dlnobs = np.log(mass2obs('disp', 1.01*M0_arr, z)/mass2obs('disp', M0_arr, z))
        if np.any(dlnobs==0.):
            if dlnobs[-1]==0: dlnobs[-1] = dlnobs[-2]
        return dlnM/dlnobs
