from __future__ import division
import numpy as np
from scipy.interpolate import interp1d, RectBivariateSpline
from scipy.stats import norm

class MisCentering(object):

    def __init__(self, kind, tau=None, fmis=None, SPT_kappa=1, SPT_beam=1.3, r200c=None):
        assert kind in ['redmapper', 'r200', 'SPT'], 'unexpected kind %s, kind must be redmapper, r200, SPT'%kind
        self.kind = kind
        # redmapper miscentering
        self.tau = tau
        self.fmis = fmis
        # SPT miscentering
        self.SPT_kappa = SPT_kappa
        self.SPT_beam = SPT_beam
        # Gupta model
        self.rho0 = 0.802
        self.sigma0 = .044
        self.sigma1 = .184


    def get_profile_mean_cov(self, R, profile, R_mis_max, SPT_xi=None, SPT_thetac=None, r200c=None, lamb_red=None):
        """Return mis-centered profile."""
        self.SPT_xi = SPT_xi
        self.SPT_thetac = SPT_thetac
        self.r200c = r200c
        self.lamb_red = lamb_red

        # Set up radial interpolation of profile
        profile_interp = interp1d(R, profile, kind='cubic', fill_value='extrapolate', bounds_error=False)

        # Integration variables
        theta = np.linspace(0, np.pi, 16)
        R_mis = np.linspace(0, R_mis_max, 16)

        # Miscentering distribution
        p_of_Rmis = self.get_p_of_Rmis(R_mis)

        # Delta_Sigma([R, R_mis, theta])
        R_eff = np.sqrt(R[:,None,None]**2 + R_mis[None,:,None]**2 + 2 * np.cos(theta)[None,None,:] * R[:,None,None] * R_mis[None,:,None])
        profile_theta = profile_interp(R_eff)/np.pi

        ##### Mean miscentered profile
        profile_mis = np.array([RectBivariateSpline(R_mis, theta, p_of_Rmis[:,None]*profile_theta[i]).integral(0, R_mis_max, 0, np.pi)
                                for i in range(len(R))])

        ##### Covariance matrix
        residuals = profile_theta.reshape(len(R),-1) - profile_mis[:,None]
        weights = (p_of_Rmis[:,None] * np.ones(profile_theta.shape[1:])).flatten()
        cov = np.cov(residuals, aweights=weights)

        return profile_mis, cov




    def get_p_of_Rmis(self, R_mis):
        if self.kind=='redmapper':
            return self.pRmis_redmapper(R_mis)
        elif self.kind=='r200':
            return self.pRmis_r200(R_mis)
        elif self.kind=='SPT':
            return self.pRmis_SPT(R_mis)

    def pRmis_SPT(self, R_mis):
        """SPT positional uncertainty"""
        sigma_ = np.sqrt((self.SPT_beam/60)**2 + (self.SPT_kappa*self.SPT_thetac/60)**2)/self.SPT_xi
        p = R_mis * norm.pdf(R_mis, 0, sigma_)
        A = sigma_**2 * norm.pdf(0, 0, sigma_)
        return p/A

    def pRmis_SPT_Magneticum(self, R_mis):
        """SZ error according to Magneticum (Gupta+ 16)"""
        p = x * (self.rho0*norm.pdf(x, 0, self.sigma0) + (1-self.rho0)*norm.pdf(x, 0, self.sigma1))
        A = self.rho0*self.sigma0**2 * norm.pdf(0, 0, self.sigma0) + (1-self.rho0)*self.sigma1**2 * norm.pdf(0, 0, self.sigma1)
        return p/A


    def pRmis_redmapper(self, R_mis):
        """Offset distribution from McClintock et al"""
        Rlamb1= 1.0 * (self.lamb_red/100)**0.2 # h^-1Mpc

        #pRmis1 = fmis* (Rmis[None,:]/(tau*Rlamb1[:,None])**2 * np.exp(-Rmis[None,:]/tau*Rlamb1[:,None]))
        pRmis1 = fmis* (R_mis/(self.tau*Rlamb1)**2 * np.exp(-R_mis/self.tau*Rlamb1))
        pRmis1[0]+=(1-self.fmis)

        return (pRmis1/np.sum(pRmis1))



    def pRmis_r200(self, R_mis):
        """Offset distribution with r200"""

        #pRmi2 = []
        #for i in xrange(len(gR)):
        pRm = self.fmis*(Rmis/((0.2*self.r200c)**2))*np.exp(-0.5*(Rmis/(0.2*self.r200c))**2)
        pRm[0]+=(1-self.fmis)
            #pRmi2.append(pRm)
        pRmis2 = np.array(pRm)

        return (pRmis2/np.sum(pRmis2))
