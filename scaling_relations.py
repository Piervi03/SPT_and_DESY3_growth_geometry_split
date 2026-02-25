import numpy as np
from astropy import table

import cosmo


################
def xi2zeta(xi):
    return np.sqrt(xi**2 - 3)


def zeta2xi(zeta):
    return np.sqrt(zeta**2 + 3)


def dlnzeta_dxi_given_xi(xi):
    """d(ln(zeta))/d xi = d(ln((xi^2 - 3)^0.5))/d xi = 0.5 / (xi^2 - 3) * 2*xi = xi/zeta^2"""
    return xi / (xi**2 - 3)


def dlnzeta_dxi_given_zeta(zeta):
    return zeta2xi(zeta)/zeta**2


##########################################
def lnmass2lnobs(name, lnmass, z, scaling,
                 cosmology=None,
                 cluster_ID=None,
                 lnM500_to_lnM200=None,
                 SPTfield=None,
                 SZ_Ez=True):
    """Returns ln-observable given (lnmass, z) using scaling relation."""
    if name == 'zeta':
        if SZ_Ez:
            ln_z_term = np.log(cosmo.Ez(z, cosmology)/cosmo.Ez(.6, cosmology))
        else:
            ln_z_term = np.log((1+z)/1.6)
        lnzeta = (scaling['Asz'] + np.log(SPTfield['GAMMA'])
                  + scaling['Bsz'] * (lnmass-np.log(scaling['SZmPivot']))
                  + (scaling['Csz'] + SPTfield['DELTA_CSZ']) * ln_z_term
                  + scaling['Esz'] * (lnmass-np.log(scaling['SZmPivot'])) * ln_z_term)
        # ECS correction
        if type(SPTfield) is table.row.Row:
            # We're looking at a single field (single row)
            if '_sptpol' in SPTfield['FIELD']:
                lnzeta += np.log(scaling['SPECS_calib'])
        else:
            # It's a table
            if len(SPTfield) == 1:
                # single field (but still a table)
                if '_sptpol' in SPTfield['FIELD'][0]:
                    lnzeta += np.log(scaling['SPECS_calib'])
            elif len(SPTfield) == len(lnzeta):
                # We're looking at a field per mass and z
                idx = ['_sptpol' in field for field in SPTfield['FIELD']]
                lnzeta[idx] += np.log(scaling['SPECS_calib'])
            else:
                raise ValueError("SPTfield length does not match expected lengths.")
        return lnzeta
    elif name == 'Yx':
        if scaling['YXPARAM'] == 'SPT_XVP':
            return (np.log(3.) - 2.5*np.log(cosmology['h']/.7)
                    + (1/scaling['Bx']
                       * (lnmass
                          - np.log(1e14 / .7**(3/2) / scaling['Ax'] / cosmo.Ez(z, cosmology)**scaling['Cx']))))
        elif scaling['YXPARAM'] == 'obs-mass':
            return (np.log(scaling['Ax'])
                    - 2.5 * np.log(cosmology['h']/.7)
                    + scaling['Bx'] * (lnmass - np.log(cosmology['h']/scaling['XraymPivot']))
                    + scaling['Cx'] * np.log(cosmo.Ez(z, cosmology)/cosmo.Ez(.6, cosmology)))
    elif name == 'Mgas':
        return (np.log(scaling['XraymPivot'] * scaling['Ax']) - 2.5 * np.log(cosmology['h']/.7)
                + scaling['Bx'] * (lnmass-np.log(scaling['XraymPivot']/cosmology['h']))
                + scaling['Cx'] * np.log(cosmo.Ez(z, cosmology)/cosmo.Ez(.6, cosmology))
                + scaling['Ex'] * ((lnmass - np.log(scaling['XraymPivot']/cosmology['h']))
                                   * np.log(cosmo.Ez(z, cosmology)/cosmo.Ez(.6, cosmology))))
    elif name == 'disp':
        h70z = cosmology['h']/.7*cosmo.Ez(z, cosmology)
        lnM200c = lnM500_to_lnM200(z, lnmass)
        if len(lnM200c) == 1:
            lnM200c = lnM200c[0]
        return (np.log(scaling['Adisp']) + (1/scaling['Bdisp'])*(lnM200c-np.log(1e15/cosmology['h']))
                + scaling['Cdisp']*np.log(h70z))
    elif name == 'richness_base':
        return (scaling['Arichness']
                + scaling['Brichness']*(lnmass-np.log(scaling['richmPivot']))
                + scaling['Crichness']*np.log((1+z)/1.6))
    elif name == 'richness_ext':
        return (scaling['Arichness_ext']
                + scaling['Brichness_ext']*(lnmass-np.log(scaling['richmPivot']))
                + scaling['Crichness_ext']*np.log((1+z)/1.6))
    elif name == 'richness':
        z_arr = np.atleast_1d(z)
        A = np.full(z_arr.shape, scaling['Arichness'])
        A[z_arr >= scaling['z_DESWISE']] = scaling['Arichness_ext']
        B = np.full(z_arr.shape, scaling['Brichness'])
        B[z_arr >= scaling['z_DESWISE']] = scaling['Brichness_ext']
        C = np.full(z_arr.shape, scaling['Crichness'])
        C[z_arr >= scaling['z_DESWISE']] = scaling['Crichness_ext']
        if np.isscalar(z):
            A = A[0]
            B = B[0]
            C = C[0]
        return A + B*(lnmass-np.log(scaling['richmPivot'])) + C*np.log((1+z)/1.6)
    elif name == 'WLMegacam':
        return np.log(scaling['bWL_Megacam']) + lnmass
    elif name == 'WLHST':
        return np.log(scaling['bWL_HST'][cluster_ID]) + lnmass
    elif name == 'WLDES':
        b_mean = np.interp(z, scaling['DES_zpivs'], scaling['DES_mean_b'])
        pc1 = np.interp(z, scaling['DES_zpivs'], scaling['DES_deltab_pc1'])
        pc2 = np.interp(z, scaling['DES_zpivs'], scaling['DES_deltab_pc2'])
        b_m = scaling['DES_bias_slope'][0] + scaling['DES_b_dev_m']*scaling['DES_bias_slope'][1]
        return (b_mean + scaling['DES_b_dev_1']*pc1 + scaling['DES_b_dev_2']*pc2
                + b_m*lnmass + np.log(scaling['DES_m_piv'])*(1-b_m))
    elif name == 'WLEuclid':
        b_mean = np.interp(z, scaling['Euclid_zpivs'], scaling['Euclid_mean_b'])
        pc = np.interp(z, scaling['Euclid_zpivs'], scaling['Euclid_deltab_pc'])
        b_m = scaling['Euclid_bias_slope'][0] + scaling['Euclid_b_dev_m']*scaling['Euclid_bias_slope'][1]
        return (b_mean + scaling['Euclid_b_dev']*pc + b_m*lnmass
                + np.log(scaling['Euclid_m_piv'])*(1-b_m))
    else:
        raise ValueError("Observable not known:", name)


#####################################
def obs2lnmass(name, obs, z, scaling,
               cosmology=None,
               cluster_ID=None,
               SPTfield=None,
               SZ_Ez=True):
    """Return lnmass given observable and z."""
    if name == 'zeta':
        if SZ_Ez:
            ln_z_term = np.log(cosmo.Ez(z, cosmology)/cosmo.Ez(.6, cosmology))
        else:
            ln_z_term = np.log((1+z)/1.6)
        tmp = (np.log(obs) - scaling['Asz'] - np.log(SPTfield['GAMMA'])
               - (scaling['Csz'] + SPTfield['DELTA_CSZ']) * ln_z_term)
        # ECS correction
        if type(SPTfield) is table.row.Row:
            # We're looking at a single field (single row)
            if '_sptpol' in SPTfield['FIELD']:
                tmp -= np.log(scaling['SPECS_calib'])
        else:
            # It's a table
            if len(SPTfield) == 1:
                # single field (but still a table)
                if '_sptpol' in SPTfield['FIELD'][0]:
                    tmp -= np.log(scaling['SPECS_calib'])
            elif len(SPTfield) == len(tmp):
                # We're looking at a field per mass and z
                idx = ['_sptpol' in field for field in SPTfield['FIELD']]
                tmp[idx] -= np.log(scaling['SPECS_calib'])
            else:
                raise ValueError("SPTfield length does not match expected lengths.")
        return np.log(scaling['SZmPivot']) + tmp / (scaling['Bsz'] + scaling['Esz']*ln_z_term)
    elif name == 'richness_base':
        lnmass = (np.log(scaling['richmPivot']) + (1/scaling['Brichness'])*(np.log(obs)
                  - scaling['Arichness'] - scaling['Crichness']*np.log((1+z)/1.6)))
        return lnmass
    elif name == 'richness_ext':
        lnmass = (np.log(scaling['richmPivot']) + (1/scaling['Brichness_ext'])*(np.log(obs)
                  - scaling['Arichness_ext'] - scaling['Crichness_ext']*np.log((1+z)/1.6)))
        return lnmass
    elif name == 'WLMegacam':
        return np.log(obs/scaling['bWL_Megacam'])
    elif name == 'WLHST':
        return np.log(obs/scaling['bWL_HST'][cluster_ID])
    elif name == 'WLDES':
        b_mean = np.interp(z, scaling['DES_zpivs'], scaling['DES_mean_b'])
        pc1 = np.interp(z, scaling['DES_zpivs'], scaling['DES_deltab_pc1'])
        pc2 = np.interp(z, scaling['DES_zpivs'], scaling['DES_deltab_pc2'])
        b_m = scaling['DES_bias_slope'][0] + scaling['DES_b_dev_m']*scaling['DES_bias_slope'][1]
        return ((np.log(obs) - b_mean - scaling['DES_b_dev_1']*pc1 - scaling['DES_b_dev_2']*pc2
                - np.log(scaling['DES_m_piv'])*(1-b_m))/b_m)
    elif name == 'WLEuclid':
        b_mean = np.interp(z, scaling['Euclid_zpivs'], scaling['Euclid_mean_b'])
        pc = np.interp(z, scaling['Euclid_zpivs'], scaling['Euclid_deltab_pc'])
        b_m = scaling['Euclid_bias_slope'][0] + scaling['Euclid_b_dev_m']*scaling['Euclid_bias_slope'][1]
        return ((np.log(obs) - b_mean - scaling['Euclid_b_dev']*pc
                - np.log(scaling['Euclid_m_piv'])*(1-b_m))/b_m)
    else:
        raise ValueError("Observable not known:", name)


##############################
def dlnM_dlnobs(name, scaling,
                cosmology=None,
                M0_arr=None,
                z=None):
    """Returns dlnM/dln(obs) for a given observable."""
    if name == 'zeta':
        return 1/scaling['Bsz']
    elif name == 'richness_base':
        return 1/scaling['Brichness']
    elif name == 'richness_ext':
        return 1/scaling['Brichness_ext']
    elif name == 'richness':
        z_arr = np.atleast_1d(z)
        B_ = np.full(z_arr.shape, scaling['Brichness'])
        B_[z_arr >= scaling['z_DESWISE']] = scaling['Brichness_ext']
        B = B_[0] if np.isscalar(z) else B_
        return 1/B
    elif name == 'Yx':
        if scaling['YXPARAM'] == 'SPT_XVP':
            return 1/(1/scaling['Bx'] - scaling['dlnMg_dlnr']/3)
        elif scaling['YXPARAM'] == 'obs-mass':
            return 1/(scaling['Bx'] - scaling['dlnMg_dlnr']/3)
    elif name == 'Mgas':
        return 1/(scaling['Bx'] - scaling['dlnMg_dlnr']/3)
    elif (name == 'WLMegacam') | (name == 'WLHST'):
        return 1.
    elif name == 'WLDES':
        b_m = scaling['DES_bias_slope'][0] + scaling['DES_b_dev_m']*scaling['DES_bias_slope'][1]
        return 1/b_m
    elif name == 'WLEuclid':
        b_m = scaling['Euclid_bias_slope'][0] + scaling['Euclid_b_dev_m']*scaling['Euclid_bias_slope'][1]
        return 1/b_m
    elif name == 'disp':
        dlnM = np.log(1.01)
        dlnobs = lnmass2lnobs('disp', np.log(1.01*M0_arr), z)-lnmass2lnobs('disp', np.log(M0_arr), z)
        if np.any(dlnobs == 0.):
            if dlnobs[-1] == 0:
                dlnobs[-1] = dlnobs[-2]
        return dlnM/dlnobs


###########################################
def richnessscatter(z, scaling):
    """Returns the scatter in richness for a given redshift."""
    if z < scaling['z_DESWISE']:
        return scaling['Drichness']
    else:
        return scaling['Drichness_ext']


def WLscatter(name, lnmass, z, scaling):
    if name == 'WLDES':
        s_mean = np.interp(z, scaling['DES_zpivs'], scaling['DES_mean_lnsimga2'])
        s_std = np.interp(z, scaling['DES_zpivs'], scaling['DES_delta_lnsigma2'])
        s_m = scaling['DES_lnsigma2_slope'][0] + scaling['DES_s_dev_m']*scaling['DES_lnsigma2_slope'][1]
        lnvar = s_mean + scaling['DES_s_dev']*s_std + s_m*(lnmass-np.log(scaling['DES_m_piv']))
    elif name == 'WLEuclid':
        s_mean = np.interp(z, scaling['Euclid_zpivs'], scaling['Euclid_mean_lnsigma2'])
        s_std = np.interp(z, scaling['Euclid_zpivs'], scaling['Euclid_delta_lnsigma2'])
        s_m = scaling['Euclid_lnsigma2_slope'][0] + scaling['Euclid_s_dev_m']*scaling['Euclid_lnsigma2_slope'][1]
        lnvar = s_mean + scaling['Euclid_s_dev']*s_std + s_m*(lnmass-np.log(scaling['Euclid_m_piv']))
    return np.exp(.5 * lnvar)
