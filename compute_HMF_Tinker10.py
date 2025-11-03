import numpy as np
from scipy.interpolate import interp1d
import cosmo


class HMFCalculator:
    def __init__(self, Deltacrit, z_arr, M_arr):
        """Initialize Tinker parameter interpolation functions."""
        self.Deltacrit = Deltacrit
        self.z_arr = z_arr
        self.M_arr = M_arr
        # Initialize Tinker interpolation
        x = np.log([200., 300., 400., 600., 800., 1200., 1600., 2400., 3200.])
        y = [.368, .363, .385, .389, .393, .365, .379, .355, .327]
        self.interp_alpha = interp1d(x, y, kind='cubic', bounds_error=False, fill_value=(y[0], y[-1]), assume_sorted=True)
        y = [.589, .585, .544, .543, .564, .623, .637, .673, .702]
        self.interp_beta = interp1d(x, y, kind='cubic', bounds_error=False, fill_value=(y[0], y[-1]), assume_sorted=True)
        y = [.864, .922, .987, 1.09, 1.20, 1.34, 1.50, 1.68, 1.81]
        self.interp_gamma = interp1d(x, y, kind='cubic', bounds_error=False, fill_value=(y[0], y[-1]), assume_sorted=True)
        y = [-.729, -.789, -.910, -1.05, -1.20, -1.26, -1.45, -1.50, -1.49]
        self.interp_phi = interp1d(x, y, kind='cubic', bounds_error=False, fill_value=(y[0], y[-1]), assume_sorted=True)
        y = [-.243, -.261, -.261, -.273, -.278, -.301, -.301, -.319, -.336]
        self.interp_eta = interp1d(x, y, kind='cubic', bounds_error=False, fill_value=(y[0], y[-1]), assume_sorted=True)

    def compute_HMF(self, cosmology, z, k, Pk):
        """Compute Tinker HMF and apply redshift volume."""
        rho_m = (cosmology['Omega_m'] - cosmology['Omega_nu']) * cosmo.RHOCRIT
        Deltamean = self.Deltacrit / cosmo.Omega_m_z(self.z_arr, cosmology)
        # Window functions [M_arr, k]
        R = (3 * self.M_arr / (4 * np.pi * rho_m))**(1/3)
        kR = k[None, :] * R[:, None]
        window = 3 * (np.sin(kR)/kR**3 - np.cos(kR)/kR**2)
        dwindow = 3/kR**4 * (3*kR*np.cos(kR) + ((kR**2 - 3)*np.sin(kR)))
        # Integrands [z_arr, M_arr, k]
        integrand_sigma2 = Pk[:, None, :] * window[None, :, :]**2 * k[None, None, :]**3
        integrand_dsigma2dM = Pk[:, None, :] * window[None, :, :] * dwindow[None, :, :] * k[None, None, :]**4
        # Sigma^2 and dsigma^2/dM [z_arr, M_arr]
        sigma2 = .5/np.pi**2 * np.trapezoid(integrand_sigma2, np.log(k), axis=-1)
        dsigma2dM = np.pi**-2 * R[None, :]/self.M_arr[None, :]/3 * np.trapezoid(integrand_dsigma2dM, np.log(k), axis=-1)
        sigma2_fine = np.exp(interp1d(z, np.log(sigma2), axis=0)(self.z_arr))
        dsigma2dM_fine = -np.exp(interp1d(z, np.log(-dsigma2dM), axis=0)(self.z_arr))
        # Tinker multiplicity function [z_arr, M_arr]
        alpha, beta, phi, eta, gamma = np.array([self.Tinker_params(self.z_arr[i], Deltamean[i])
                                                 for i in range(len(self.z_arr))]).T
        nu = 1.686/np.sqrt(sigma2_fine)
        fsigma = nu * alpha[:, None] * (1 + (beta[:, None]*nu)**(-2.*phi[:, None])) * nu**(2*eta[:, None]) * np.exp(-gamma[:, None]*nu**2/2.)
        dNdlnM_noVol = - fsigma * rho_m * dsigma2dM_fine/2/sigma2_fine
        # Apply redshift volume
        deltaV = cosmo.deltaV(self.z_arr, cosmology)
        dNdlnM = dNdlnM_noVol * deltaV[:, None]
        return dNdlnM_noVol, dNdlnM

    def Tinker_params(self, z, Deltamean):
        """For given redshift and mean overdensity, return the five Tinker
        parameters."""
        lnDeltamean = np.log(Deltamean)
        alpha = self.interp_alpha(lnDeltamean)
        beta = self.interp_beta(lnDeltamean) * (1+z)**.20
        phi = self.interp_phi(lnDeltamean) * (1+z)**-.08
        eta = self.interp_eta(lnDeltamean) * (1+z)**.27
        gamma = self.interp_gamma(lnDeltamean) * (1+z)**-.01
        return alpha, beta, phi, eta, gamma
