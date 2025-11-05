import numpy as np

import cosmo


def generic_miscenter(params, lam, z):
    """Return parameters of generic intrinsic miscentering in units [Mpc/h]."""
    alpha = params['alpha_0'] * ((1+z)/1.5)**params['alpha_z'] * (lam/60)**params['alpha_lam']
    mis_phys_0 = params['comp0_0'] * ((1+z)/1.5)**params['comp0_z'] * (lam/60)**params['comp0_lam']
    Delta_mis_phys = params['comp1_0'] * ((1+z)/1.5)**params['comp1_z'] * (lam/60)**params['comp1_lam']
    mis_phys_1 = mis_phys_0 + Delta_mis_phys
    return alpha, mis_phys_0, mis_phys_1


class MisCentering(object):

    def __init__(self, opt):
        assert opt['kind'] in ['G21', 'optical', 'MCMF', 'SPT'], "unexpected kind %s, kind must be G21, optical, MCMF, SPT" % opt['kind']
        self.opt = opt

    def get_mean_Rmis(self, cluster, cosmology=None):
        if self.opt['kind'] in ['optical', 'MCMF']:
            return self.get_mean_Rmis_optical(cluster)
        elif self.opt['kind'] == 'SPT':
            return self.get_mean_Rmis_SPT(cluster, cosmology)
        elif self.opt['kind'] == 'G21':
            return self.get_mean_Rmis_G21(cluster)

    def get_mean_Rmis_G21(self, cluster):
        """Mean miscentering [Mpc/h] for the generic model in Grandis+21."""
        R = ((cluster['richness']/100)**.2 * np.sqrt(np.pi/2)
             * (self.opt['rho']*self.opt['sigma0'] + (1-self.opt['rho'])*self.opt['sigma1']))
        return R

    def get_mean_Rmis_optical(self, cluster):
        """Mean miscentering [Mpc/h] for optical centers."""
        rho, sigma_0, sigma_1 = generic_miscenter(self.opt, cluster['richness'], cluster['REDSHIFT'])
        R = np.sqrt(np.pi/2) * (rho*sigma_0 + (1-rho)*sigma_1)
        return R

    def get_mean_Rmis_SPT(self, cluster, cosmology):
        """Mean miscentering, accounting for SPT positional uncertainty and
        intrinsic SZ miscentering."""
        sigma_obs_arcmin = np.sqrt((1.3**2 + (5./60)**2
                                   + self.opt['kappa_SPT']**2 * cluster['THETA_CORE']**2)/cluster['XI']**2)
        sigma_obs_Mpch = sigma_obs_arcmin / 60 * np.pi/180 * cosmo.dA(cluster['REDSHIFT'], cosmology)
        rho, sigma_0, sigma_1 = generic_miscenter(self.opt, cluster['richness'], cluster['REDSHIFT'])
        sigma_0 = np.sqrt(sigma_0**2 + sigma_obs_Mpch**2)
        sigma_1 = np.sqrt(sigma_1**2 + sigma_obs_Mpch**2)
        R = np.sqrt(np.pi/2) * (rho*sigma_0 + (1-rho)*sigma_1)
        return R
