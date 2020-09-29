from __future__ import division, print_function
import numpy as np
from scipy.interpolate import interp1d, InterpolatedUnivariateSpline, RectBivariateSpline
from scipy.stats import norm, rayleigh

import cosmo

class MisCentering(object):

    def __init__(self, kind, tau=None, fmis=None,
                 # SPT_kappa=1, SPT_beam=1.3,
                 sigma_SPT=None,
                 r200c=None):
        assert kind in ['redmapper', 'r200', 'SPT', 'arcmin'], "unexpected kind %s, kind must be redmapper, r200, SPT, arcmin"%kind
        self.kind = kind

        # SPT miscentering with double Rayleigh
        self.theta_beam = 1.3
        self.rho0 = 0.769
        self.sigma0 = 0.043
        self.sigma1 = 0.184


    def get_Rmis_extr_opt(self, lam, z, miscenter_opt, cosmology):

        # lam = miscenter_opt['A_lambda'] * (input_M_arr/3.e14)**miscenter_opt['B_lambda'] * \
        #       (cosmo.Ez(z, cosmology)/cosmo.Ez(.6, cosmology))**miscenter_opt['C_lambda']

        sigma0 = miscenter_opt['sigma0'] * ((1+lam)/60)**miscenter_opt['sigma0_lam'] * ((1+z)/1.6)**miscenter_opt['sigma0_z']
        sigma1 = miscenter_opt['sigma1'] * ((1+lam)/60)**miscenter_opt['sigma1_lam']
        mis = miscenter_opt['rho']*sigma0 + (1-miscenter_opt['rho'])*sigma1

        return mis


    def get_mean_Rmis_SPT(self, r_Delta, r_core, xi, dA):
        """Mean off-centering, accounting for SPT positional uncertainty and
        intrinsic SZ miscentering."""
        r_beam = self.theta_beam * dA * np.pi/180/60
        var_SPT = (r_beam**2 + r_core**2)/xi**2
        sigma0_tot = np.sqrt((r_Delta*self.sigma0)**2 + var_SPT)
        sigma1_tot = np.sqrt((r_Delta*self.sigma1)**2 + var_SPT)
        mean_mis = np.sqrt(np.pi/2) * (self.rho0 * sigma0_tot + (1-self.rho0) * sigma1_tot)
        return mean_mis


    def get_profile_mean_draws(self, R, profile, R_mis_max,
                               r500_deg=None,
                               sigma_SPT=None,
                               r200c=None, lamb_red=None):
        # """Return mis-centered profile."""
        """Return `draws` and `weights` of the mis-centered profile.
        Normalization of the weights is such that the mean profile is simply the
        weighted sum of the draws."""
        self.r200c = r200c
        self.lamb_red = lamb_red
        self.r500_deg = r500_deg
        self.sigma_SPT = sigma_SPT

        # Set up radial interpolation of profile
        profile_interp = interp1d(R, profile, kind='cubic', fill_value='extrapolate', bounds_error=False)

        # Integration variables
        theta = np.linspace(0, np.pi, self.len_theta)
        # R_mis = np.insert(np.logspace(np.log10(R_mis_max)-2.5, np.log10(R_mis_max), self.len_Rmis-1), 0, 0)
        R_mis = np.linspace(0, R_mis_max, self.len_Rmis)
        self.R_mis = R_mis

        # Miscentering distribution
        p_of_Rmis = self.get_p_of_Rmis(R_mis)

        # Delta_Sigma([R, R_mis, theta])
        R_eff = np.sqrt(R[:,None,None]**2 + R_mis[None,:,None]**2 + 2*np.cos(theta)[None,None,:]*R[:,None,None]*R_mis[None,:,None])
        profile_theta = profile_interp(R_eff)/np.pi
        self.profile_theta = profile_theta

        profile_Rmis = np.trapz(profile_theta, theta, axis=-1)

        # Mean miscentered profile
        # profile_mis = np.array([RectBivariateSpline(R_mis, theta, p_of_Rmis[:,None]*profile_theta[i]).integral(0, R_mis_max, 0, np.pi)
        #                        for i in range(len(R))])

        # Draws from P(R_mis)
        # weights = .5*(p_of_Rmis[:-1]+p_of_Rmis[1:]) * np.diff(R_mis)
        # weights = np.diff(R_mis)
        # profile_Rmis_trapz = .5*(p_of_Rmis[:-1]*profile_Rmis[:,:-1] + p_of_Rmis[1:]*profile_Rmis[:,1:])

        return profile_Rmis, R_mis, p_of_Rmis




    def get_p_of_Rmis(self, R_mis):
        if self.kind=='redmapper':
            return self.pRmis_redmapper(R_mis)
        elif self.kind=='r200':
            return self.pRmis_r200(R_mis)
        elif self.kind=='SPT':
            return self.pRmis_SPT(R_mis)
        elif self.kind=='arcmin':
            return self.pRmis_arcmin(R_mis)


    def pRmis_SPT(self, R_mis):
        """Convolution of a double Rayleigh function with the SPT positional
        uncertainty."""
        #p_SPT = R_mis * norm.pdf(R_mis, 0, self.sigma_SPT) / (self.sigma_SPT/np.sqrt(2*np.pi))
        #self.p_SPT = p_SPT
        # Double Rayleigh function convolved with SPT positional uncertainty
        x = R_mis/self.r500_deg
        sigma0 = np.sqrt(self.sigma0**2 + (self.sigma_SPT/self.r500_deg)**2)
        sigma1 = np.sqrt(self.sigma1**2 + (self.sigma_SPT/self.r500_deg)**2)

        res = x * np.sqrt(2*np.pi) * (
            self.rho0/sigma0 * norm.pdf(x, 0, sigma0) \
            + (1-self.rho0)/sigma1 * norm.pdf(x, 0, sigma1))

        return res


    def pRmis_redmapper(self, R_mis):
        """Offset distribution from McClintock et al"""
        Rlamb1= 1.0 * (self.lamb_red/100)**0.2 # h^-1Mpc

        #pRmis1 = fmis* (Rmis[None,:]/(tau*Rlamb1[:,None])**2 * np.exp(-Rmis[None,:]/tau*Rlamb1[:,None]))
        pRmis1 = fmis* (R_mis/(self.tau*Rlamb1)**2 * np.exp(-R_mis/self.tau*Rlamb1))
        pRmis1[0]+=(1-self.fmis)

        return (pRmis1/np.sum(pRmis1))



    def pRmis_r200(self, R_mis):
        """Offset distribution with r200"""
        x = R_mis/self.r200c
        res = self.rho0 * rayleigh.pdf(x, scale=self.sigma0) + (1-self.rho0) * rayleigh.pdf(x, scale=self.sigma1)

        return res

    def pRmis_arcmin(self, R_mis):
        """Offset distribution [arcmin]"""
        return self.rho0 * rayleigh.pdf(R_mis, scale=self.sigma0) + (1-self.rho0) * rayleigh.pdf(R_mis, scale=self.sigma1)
