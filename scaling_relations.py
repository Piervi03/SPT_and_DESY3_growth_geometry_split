from __future__ import division
import numpy as np

import cosmo

####################
##### xi--zeta relations
def xi2zeta(xi):
    return np.sqrt(xi**2 - 3)
def zeta2xi(zeta):
    return np.sqrt(zeta**2 + 3)
def dlnzeta_dxi_given_xi(xi):
    """d(ln(zeta))/d xi = d(ln((xi^2 - 3)^0.5))/d xi = 0.5 / (xi^2 - 3) * 2*xi = xi/zeta^2"""
    return xi / (xi**2 - 3)
def dlnzeta_dxi_given_zeta(zeta):
    return zeta2xi(zeta)/zeta**2


####################
def mass2obs(name, mass, z, scaling, cosmology=None, cluster_ID=None):
    """Returns observable given (mass, z) using scaling relation."""
    if name=='zeta':
        lnzeta = scaling['Asz'] + scaling['Bsz'] * np.log(mass/scaling['SZmPivot'])\
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
            * ((1+z)/1.6)**scaling['Crichness']
    elif name=='WLMegacam':
        return scaling['bWL_Megacam'] * mass
    elif name=='WLHST':
        return scaling['bWL_HST'][cluster_ID] * mass
    elif name=='WLDES':
        bias_z = scaling['DESwl_bias_mean'] + np.array([scaling['DES_b_dev_%d'%i] for i in range(3)])*scaling['DESwl_bias_std']
        b = np.interp(z, scaling['DESwl_z'], bias_z)
        return scaling['DES_m_piv'] * np.exp(b)  * (mass/scaling['DES_m_piv'])**scaling['DES_b_m']
    else:
        raise ValueError("Observable not known:", name)


####################
def obs2mass(name, obs, z, scaling, cosmology=None, cluster_ID=None):
    """Return mass given observable and z."""
    if name=='zeta':
        ln_M_M0 = (np.log(obs) - scaling['Asz'] - scaling['Csz']*np.log(cosmo.Ez(z, cosmology)/cosmo.Ez(.6, cosmology)))\
                  / (scaling['Bsz'] + scaling['Esz']*np.log(cosmo.Ez(z, cosmology)/cosmo.Ez(.6, cosmology)))
        mass = scaling['SZmPivot'] * np.exp(ln_M_M0)
        return mass
    elif name=='richness':
        mass = scaling['richmPivot'] * (obs / scaling['Arichness'] / ((1+z)/1.6)**scaling['Crichness']) ** (1/scaling['Brichness'])
        return mass
    elif name=='WLMegacam':
        return obs/scaling['bWL_Megacam']
    elif name=='WLHST':
        return obs/scaling['bWL_HST'][cluster_ID]
    elif name=='WLDES':
        bias_z = scaling['DESwl_bias_mean'] + np.array([scaling['DES_b_dev_%d'%i] for i in range(3)])*scaling['DESwl_bias_std']
        b = np.interp(z, scaling['DESwl_z'], bias_z)
        return scaling['DES_m_piv'] * (obs / scaling['DES_m_piv'] / np.exp(b))**(1/scaling['DES_b_m'])
    else:
        raise ValueError("Observable not known:", name)


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
    elif (name=='WLMegacam')|(name=='WLHST'):
        return 1.
    elif name=='WLDES':
        return 1/scaling['DES_b_m']
    elif name=='disp':
        dlnM = np.log(1.01)
        dlnobs = np.log(mass2obs('disp', 1.01*M0_arr, z)/mass2obs('disp', M0_arr, z))
        if np.any(dlnobs==0.):
            if dlnobs[-1]==0: dlnobs[-1] = dlnobs[-2]
        return dlnM/dlnobs


####################
def WLscatter(name, mass, z, scaling):
    if name=='main':
        scatter_z = scaling['DESwl_scatter_mean'] + np.array([scaling['DES_s_dev_%d'%i] for i in range(3)])*scaling['DESwl_scatter_std']
        s = np.interp(z, scaling['DESwl_z'], scatter_z)
        lnvar = s + scaling['DES_s_M']*np.log(mass/scaling['DES_m_piv'])
        return  np.exp(.5 * lnvar)
    elif name=='wide':
        return np.sqrt(np.exp(scaling['DES_wide_s_0'] + scaling['DES_wide_s_1']*(mass/scaling['DES_m_piv'])))
