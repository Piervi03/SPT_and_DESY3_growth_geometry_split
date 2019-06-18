from __future__ import division
import numpy as np
from scipy.interpolate import interp1d, InterpolatedUnivariateSpline, RectBivariateSpline
from scipy.stats import norm, rayleigh

class MisCentering(object):

    def __init__(self, kind, tau=None, fmis=None,
                 # SPT_kappa=1, SPT_beam=1.3,
                 sigma_SPT=None,
                 r200c=None):
        assert kind in ['redmapper', 'r200', 'SPT'], 'unexpected kind %s, kind must be redmapper, r200, SPT'%kind
        self.kind = kind
        # redmapper miscentering
        # self.tau = tau
        # self.fmis = fmis

        # SPT miscentering with double Rayleigh
        self.rho0 = 0.802
        self.sigma0 = .044
        self.sigma1 = .184
        # self.rho0 = 0.63
        # self.sigma0 = .07
        # self.sigma1 = .25

        self.len_theta = 8
        self.len_Rmis = 16


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
        R_mis = np.insert(np.logspace(np.log10(R_mis_max)-2.5, np.log10(R_mis_max), self.len_Rmis-1), 0, 0)
        self.R_mis = R_mis

        # Miscentering distribution
        p_of_Rmis = self.get_p_of_Rmis(R_mis)

        # Delta_Sigma([R, R_mis, theta])
        R_eff = np.sqrt(R[:,None,None]**2 + R_mis[None,:,None]**2 + 2*np.cos(theta)[None,None,:]*R[:,None,None]*R_mis[None,:,None])
        profile_theta = profile_interp(R_eff)/np.pi
        self.profile_theta = profile_theta

        profile_Rmis = np.trapz(profile_theta, theta, axis=-1)

        # Mean miscentered profile
        profile_mis = None#np.array([RectBivariateSpline(R_mis, theta, p_of_Rmis[:,None]*profile_theta[i]).integral(0, R_mis_max, 0, np.pi)
        #                        for i in range(len(R))])

        # Draws from P(R_mis)
        weights = .5*(p_of_Rmis[:-1]+p_of_Rmis[1:]) * np.diff(R_mis)
        profile_Rmis_trapz = .5*(profile_Rmis[:,:-1]+profile_Rmis[:,1:])

        return profile_mis, profile_Rmis_trapz, weights




    def get_p_of_Rmis(self, R_mis):
        if self.kind=='redmapper':
            return self.pRmis_redmapper(R_mis)
        elif self.kind=='r200':
            return self.pRmis_r200(R_mis)
        elif self.kind=='SPT':
            return self.pRmis_SPT(R_mis)


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
