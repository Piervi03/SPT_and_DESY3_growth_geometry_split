from __future__ import division
import numpy as np
from scipy.interpolate import interp1d
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


    def get_profile_mean_cov(self, R, Delta_Sigma, R_mis_max, SPT_xi=None, SPT_thetac=None, r200c=None, lamb_red=None):
        """Return mis-centered Delta_Sigma profile."""
        self.SPT_xi = SPT_xi
        self.SPT_thetac = SPT_thetac
        self.r200c = r200c
        self.lamb_red = lamb_red

        # Set up radial interpolation of Delta_Sigma
        Delta_Sigma_interp = interp1d(R, Delta_Sigma, kind='cubic', fill_value=0, bounds_error=False)

        # Integration variables
        theta = np.linspace(0, 2*np.pi, 32)
        R_mis = np.linspace(0, R_mis_max, 64)

        # Miscentering distribution
        p_of_Rmis = self.get_p_of_Rmis(R_mis)

        # Delta_Sigma([R, R_mis, theta])
        R_eff = np.sqrt(R[:,None,None]**2 + R_mis[None,:,None]**2 + 2 * np.cos(theta)[None,None,:] * R[:,None,None] * R_mis[None,:,None])
        Delta_Sigma_theta = Delta_Sigma_interp(R_eff)

        ##### Mean miscentered profile
        # Integrate over theta
        # [R_mis, R]
        Delta_Sigma_Rmis = np.trapz(Delta_Sigma_theta, theta, axis=-1)
        # Integrate over R_mis
        Delta_Sigma_mis = np.trapz(Delta_Sigma_Rmis * p_of_Rmis[None,:], R_mis, axis=-1)


        ##### Covariance matrix
        residuals = Delta_Sigma_theta.reshape(len(R),-1) - Delta_Sigma_mis[:,None]
        weights = (p_of_Rmis[:,None] * np.ones(Delta_Sigma_theta.shape[1:])).flatten()
        cov = np.cov(residuals, aweights=weights)

        return Delta_Sigma_mis, cov




    def get_p_of_Rmis(self, R_mis):
        if self.kind=='redmapper':
            return self.pRmis_redmapper(R_mis)
        elif self.kind=='r200':
            return self.pRmis_r200(R_mis)
        elif self.kind=='SPT':
            return self.pRmis_SPT(R_mis)

    def pRmis_SPT(self, R_mis):
        """SPT miscentering"""
        sigma_ = np.sqrt((self.SPT_beam/60)**2 + (self.SPT_kappa*self.SPT_thetac/60)**2)/self.SPT_xi
        p = norm.pdf(R_mis, 0, sigma_)
        return p


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
