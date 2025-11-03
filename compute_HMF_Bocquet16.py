import numpy as np
from scipy.interpolate import interp1d
import cosmo


class HMFCalculator:
    def __init__(self, Deltacrit, z_arr, M_arr):
        assert Deltacrit in [200., 500.], "Deltacrit must be 200. or 500., not %f" % Deltacrit
        self.Deltacrit = Deltacrit
        self.z_arr = z_arr
        self.M_arr = M_arr

    def get_factors(self, Omega_m, z):
        """Return the universality correction factors."""
        if self.Deltacrit == 200.:
            gamma0 = 3.54e-2 + Omega_m**.09
            gamma1 = 4.56e-2 + 2.68e-2/Omega_m
            gamma2 = 0.721 + 3.5e-2/Omega_m
            gamma3 = .628 + .164/Omega_m
            delta0 = -1.67e-2 + 2.18e-2*Omega_m
            delta1 = 6.52e-3 - 6.86e-3*Omega_m
            alpha = gamma0 + gamma1 * np.exp(-((gamma2-z)/gamma3)**2)
            beta = delta0 + delta1*z
        elif self.Deltacrit == 500.:
            alpha0 = .88 + .329*Omega_m
            alpha1 = 1. + 4.31e-2/Omega_m
            alpha2 = -.365 + .254/Omega_m
            alpha = alpha0 * (alpha1*z+alpha2)/(z+alpha2)
            beta = -1.7e-2 + 3.74e-3*Omega_m
        return alpha, beta

    def get_params(self, z):
        """Return the mass function parameters as function of redshift."""
        if self.Deltacrit == 200.:
            z0params = np.array([.222, 1.71, 2.24, 1.46])
            zparams = np.array([.269, .321, -.621, -.153])
        elif self.Deltacrit == 500.:
            z0params = np.array([.241, 2.18, 2.35, 2.02])
            zparams = np.array([.370, .251, -.698, -.310])
        return z0params[:, None] * (1+z[None, :])**zparams[:, None]

    def compute_HMF(self, cosmology, z, k, Pk):
        """Compute Bocquet et al. (2016) HMF and apply redshift volume."""
        # Setup
        rho_m = (cosmology['Omega_m'] - cosmology['Omega_nu']) * cosmo.RHOCRIT

        # Compute sigma(M)
        # Radius [M_arr]
        R = (3 * self.M_arr / (4 * np.pi * rho_m))**(1/3)
        # [M_arr, k]
        kR = k[None, :] * R[:, None]
        # Window functions [M_arr, k]
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

        # Compute HMF (unit volume) [z_arr, M_arr]
        A, a, b, c = self.get_params(self.z_arr)
        fsigma = A[:, None] * ((np.sqrt(sigma2_fine)/b[:, None])**-a[:, None] + 1) * np.exp(-c[:, None]/sigma2_fine)
        dNdlnM_noVol = - fsigma * rho_m * dsigma2dM_fine/2/sigma2_fine
        # Universality correction
        alpha, beta = self.get_factors(cosmology['Omega_m'], self.z_arr)
        dNdlnM_noVol *= alpha[:, None] + beta[:, None]*np.log(self.M_arr/cosmology['h'])[None, :]

        # Apply redshift volume
        deltaV = cosmo.deltaV(self.z_arr, cosmology)
        dNdlnM = dNdlnM_noVol * deltaV[:, None]

        # Return HMF
        return dNdlnM_noVol, dNdlnM
