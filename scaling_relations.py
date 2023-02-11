from __future__ import division
import numpy as np
from scipy.interpolate import interp1d

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
def lnmass2lnobs(name, lnmass, z, scaling, cosmology=None, cluster_ID=None, lnM500_to_lnM200=None):
    """Returns ln-observable given (lnmass, z) using scaling relation."""
    if name=='zeta':
        lnE_z_term = np.log(cosmo.Ez(z, cosmology)/cosmo.Ez(.6, cosmology))
        return (scaling['Asz']
                + scaling['Bsz'] * (lnmass-np.log(scaling['SZmPivot']))
                + scaling['Csz'] * lnE_z_term
                + scaling['Esz'] * (lnmass-np.log(scaling['SZmPivot']))*lnE_z_term)
    elif name=='Yx':
        if scaling['YXPARAM']=='SPT_XVP':
            return (np.log(3.) -2.5*np.log(cosmology['h']/.7)
                    + (1/scaling['Bx'])*(lnmass - np.log(1e14 /.7**(3/2) / scaling['Ax'] / cosmo.Ez(z, cosmology)**scaling['Cx'])))
        elif scaling['YXPARAM']=='obs-mass':
            return (np.log(scaling['Ax'])
                    -2.5 * np.log(cosmology['h']/.7)
                    + scaling['Bx'] * (lnmass - np.log(cosmology['h']/scaling['XraymPivot']))
                    + scaling['Cx']* np.log(cosmo.Ez(z, cosmology)/cosmo.Ez(.6, cosmology)))
    elif name=='Mgas':
        return (np.log(scaling['XraymPivot'] * scaling['Ax']) - 2.5 * np.log(cosmology['h']/.7)
                + scaling['Bx'] * (lnmass-np.log(scaling['XraymPivot']/cosmology['h']))
                + scaling['Cx'] * np.log(cosmo.Ez(z, cosmology)/cosmo.Ez(.6, cosmology))
                + scaling['Ex'] * (lnmass - np.log(scaling['XraymPivot']/cosmology['h']))*np.log(cosmo.Ez(z, cosmology)/cosmo.Ez(.6, cosmology)))
    elif name=='disp':
        h70z = cosmology['h']/.7*cosmo.Ez(z, cosmology)
        lnM200c = lnM500_to_lnM200(z, lnmass)
        if len(lnM200c)==1:
            lnM200c = lnM200c[0]
        return np.log(scaling['Adisp']) + (1/scaling['Bdisp'])*(lnM200c-np.log(1e15/cosmology['h'])) +scaling['Cdisp']*np.log(h70z)
    elif name=='richness':
        return (scaling['Arichness']
                + scaling['Brichness']*(lnmass-np.log(scaling['richmPivot']))
                + scaling['Crichness']*np.log((1+z)/1.6))
    elif name=='WLMegacam':
        return np.log(scaling['bWL_Megacam']) + lnmass
    elif name=='WLHST':
        return np.log(scaling['bWL_HST'][cluster_ID]) + lnmass
    elif name=='WLDES':
        bias_z = scaling['DESwl_bias_mean'] + np.array([scaling['DES_b_dev_%d'%i] for i in range(3)])*scaling['DESwl_bias_std']
        b_z = interp1d(scaling['DESwl_z'], bias_z, fill_value='extrapolate')
        b_m = scaling['DESwl_bias_m_mean'] + scaling['DES_b_dev_m']*scaling['DESwl_bias_m_std']
        return b_z(z) + b_m*lnmass + np.log(scaling['DES_m_piv'])*(1-b_m)
    else:
        raise ValueError("Observable not known:", name)


####################
def obs2lnmass(name, obs, z, scaling, cosmology=None, cluster_ID=None):
    """Return lnmass given observable and z."""
    if name=='zeta':
        lnE_z_term = np.log(cosmo.Ez(z, cosmology)/cosmo.Ez(.6, cosmology))
        return (np.log(scaling['SZmPivot'])
                + (np.log(obs) - scaling['Asz'] - scaling['Csz']*lnE_z_term) / (scaling['Bsz'] + scaling['Esz']*lnE_z_term))
    elif name=='richness':
        lnmass = np.log(scaling['richmPivot']) + (1/scaling['Brichness'])*(np.log(obs) - scaling['Arichness'] - scaling['Crichness']*np.log((1+z)/1.6))
        return lnmass
    elif name=='WLMegacam':
        return np.log(obs/scaling['bWL_Megacam'])
    elif name=='WLHST':
        return np.log(obs/scaling['bWL_HST'][cluster_ID])
    elif name=='WLDES':
        bias_z = scaling['DESwl_bias_mean'] + np.array([scaling['DES_b_dev_%d'%i] for i in range(3)])*scaling['DESwl_bias_std']
        f = interp1d(scaling['DESwl_z'], bias_z, fill_value='extrapolate')
        b = f(z)
        b_m = scaling['DESwl_bias_m_mean'] + scaling['DES_b_dev_m']*scaling['DESwl_bias_m_std']
        return np.log(scaling['DES_m_piv']) + (1/b_m)*(np.log(obs / scaling['DES_m_piv']) - b)
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
        b_m = scaling['DESwl_bias_m_mean'] + scaling['DES_b_dev_m']*scaling['DESwl_bias_m_std']
        return 1/b_m
    elif name=='disp':
        dlnM = np.log(1.01)
        dlnobs = lnmass2lnobs('disp', np.log(1.01*M0_arr), z)-lnmass2lnobs('disp', np.log(M0_arr), z)
        if np.any(dlnobs==0.):
            if dlnobs[-1]==0:
                dlnobs[-1] = dlnobs[-2]
        return dlnM/dlnobs


####################
def WLscatter(name, lnmass, z, scaling):
    if name=='main':
        scatter_z = scaling['DESwl_scatter_mean'] + np.array([scaling['DES_s_dev_%d'%i] for i in range(3)])*scaling['DESwl_scatter_std']
        s_z = interp1d(scaling['DESwl_z'], scatter_z, fill_value='extrapolate')
        s_m = scaling['DESwl_scatter_m_mean'] + scaling['DES_s_dev_m']*scaling['DESwl_scatter_m_std']
        lnvar = s_z(z) + s_m*(lnmass-np.log(scaling['DES_m_piv']))
        return np.exp(.5 * lnvar)
    elif name=='wide':
        return np.sqrt(np.exp(scaling['DES_wide_s_0'] + scaling['DES_wide_s_1']*(lnmass-np.log(scaling['DES_m_piv']))))
