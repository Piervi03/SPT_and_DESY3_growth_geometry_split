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
# def dxi_dzeta(zeta):
    # return zeta / (zeta**2 + 3)


####################
# def obs2mass(name, obs, z, scaling, cosmology):
#     """Returns mass given (observable, z) using scaling relation."""
#     if name=='zeta':
#         lnM = np.log(scaling['SZmPivot']) + (np.log(obs) - np.log(scaling['Asz'])\
#             - scaling['Csz']*np.log(cosmo.Ez(z, cosmology)/cosmo.Ez(.6, cosmology)))/(scaling['Bsz']\
#             + scaling['Esz']*np.log(cosmo.Ez(z, cosmology)/cosmo.Ez(.6, cosmology)))
#         return np.exp(lnM)
#     elif name=='Yx':
#         if scaling['YXPARAM']=='SPT_XVP':
#             return 1e14 * scaling['Ax'] * cosmology['h']**1.5\
#                 * (cosmology['h']/.72)**(2.5*scaling['Bx']-1.5)\
#                 * (obs/3.)**scaling['Bx'] * cosmo.Ez(z, cosmology)**scaling['Cx']
#         elif scaling['YXPARAM']=='Munich':
#             return scaling['XraymPivot'] * cosmology['h']**1.5 * (obs/(scaling['Ax']
#                 *(cosmo.Ez(z, cosmology)/cosmo.Ez(.6, cosmology))**scaling['Cx']))**(1/scaling['Bx'])
#     elif name=='Mgas':
#         return scaling['XraymPivot'] * cosmology['h'] * (obs/scaling['XraymPivot']/scaling['Ax']
#             /(cosmo.Ez(z, cosmology)/cosmo.Ez(.6, cosmology))**scaling['Cx'])**(1./scaling['Bx'])
#     elif name=='disp':
#         h70z = cosmology['h']/.7*cosmo.Ez(z, cosmology)
#         M200c = 1e15*cosmology['h'] * (obs/scaling['Adisp']/h70z**scaling['Cdisp'])**scaling['Bdisp']
#         return np.exp(lnM200_to_lnM500(z, np.log(M200c)))
#     elif name=='richness':
#         return scaling['richmPivot']* (obs/scaling['Arichness']
#             /(cosmo.Ez(z, cosmology)/cosmo.Ez(.6, cosmology))**scaling['Crichness'])**(1/scaling['Brichness'])
#     elif name=='WLMegacam':
#         return obs/scaling['bWL_Megacam']
#     elif name=='WLHST':
#         return obs/scaling['bWL_HST']
#     elif name=='WLDES':
#         return obs/scaling['bWL_DES']
#     else:
#         raise ValueError("Observable not known:",name)


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
            return 3*(mass*1e-14/(scaling['Ax'] * cosmology['h']**1.5
                * (cosmology['h']/.72)**(2.5*scaling['Bx']-1.5)
                * cosmo.Ez(z, cosmology)**scaling['Cx']))**(1/scaling['Bx'])
        elif scaling['YXPARAM']=='Munich':
            return scaling['Ax']* (mass/cosmology['h']**1.5/scaling['XraymPivot'])**scaling['Bx']\
                * (cosmo.Ez(z, cosmology)/cosmo.Ez(.6, cosmology))**scaling['Cx']
    elif name=='Mgas':
        lnMgas = np.log(scaling['XraymPivot'] * scaling['Ax']) + scaling['Bx']*np.log(mass/scaling['XraymPivot']/cosmology['h'])\
            + scaling['Cx']*np.log(cosmo.Ez(z, cosmology)/cosmo.Ez(.6, cosmology))\
            + scaling['Ex']*np.log(mass/scaling['XraymPivot']/cosmology['h'])*np.log(cosmo.Ez(z, cosmology)/cosmo.Ez(.6, cosmology))
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
        elif scaling['YXPARAM']=='Munich':
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
