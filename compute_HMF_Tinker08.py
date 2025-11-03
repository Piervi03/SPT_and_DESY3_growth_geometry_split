import numpy as np
from scipy.interpolate import interp1d
import cosmo


class HMFCalculator:
    def __init__(self, Deltacrit, z_arr, M_arr):
        """Initialize Tinker parameter interpolation functions."""
        self.Deltacrit = Deltacrit
        self.z_arr = z_arr
        self.M_arr = M_arr
        # Initialize Tinker interpolation (A, a, b, c)
        x = np.log((200., 300., 400., 600., 800., 1200., 1600., 2400., 3200.))
        y = (1.858659e-01, 1.995973e-01, 2.115659e-01, 2.184113e-01, 2.480968e-01, 2.546053e-01, 2.600000e-01, 2.600000e-01, 2.600000e-01)
        self.interp_A = interp1d(x, y, kind='cubic', bounds_error=False, fill_value=(y[0], y[-1]), assume_sorted=True)
        y = (1.466904e+00, 1.521782e+00, 1.559186e+00, 1.614585e+00, 1.869936e+00, 2.128056e+00, 2.301275e+00, 2.529241e+00, 2.661983e+00)
        self.interp_a = interp1d(x, y, kind='cubic', bounds_error=False, fill_value=(y[0], y[-1]), assume_sorted=True)
        y = (2.571104e+00, 2.254217e+00, 2.048674e+00, 1.869559e+00, 1.588649e+00, 1.507134e+00, 1.464374e+00, 1.436827e+00, 1.405210e+00)
        self.interp_b = interp1d(x, y, kind='cubic', bounds_error=False, fill_value=(y[0], y[-1]), assume_sorted=True)
        y = (1.193958e+00, 1.270316e+00, 1.335191e+00, 1.446266e+00, 1.581345e+00, 1.795050e+00, 1.965613e+00, 2.237466e+00, 2.439729e+00)
        self.interp_c = interp1d(x, y, kind='cubic', bounds_error=False, fill_value=(y[0], y[-1]), assume_sorted=True)

    def compute_HMF(self, cosmology, z, k, Pk):
        """Compute Tinker HMF and apply redshift volume."""
        # Setup
        rho_m = (cosmology['Omega_m'] - cosmology['Omega_nu']) * cosmo.RHOCRIT
        # Mean overdensity at each redshift
        Deltamean = self.Deltacrit / cosmo.Omega_m_z(self.z_arr, cosmology)

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

        # Compute Tinker HMF (unit volume)
        A, a, b, c = np.array([self.Tinker_params(self.z_arr[i], Deltamean[i]) for i in range(len(self.z_arr))]).T
        # HMF [z_arr, M_arr]
        fsigma = A[:, None] * ((np.sqrt(sigma2_fine)/b[:, None])**-a[:, None] + 1) * np.exp(-c[:, None]/sigma2_fine)
        dNdlnM_noVol = - fsigma * rho_m * dsigma2dM_fine/2/sigma2_fine

        # Apply redshift volume
        deltaV = cosmo.deltaV(self.z_arr, cosmology)
        dNdlnM = dNdlnM_noVol * deltaV[:, None]

        # Return HMF
        return dNdlnM_noVol, dNdlnM

    def Tinker_params(self, z, Deltamean):
        """For given redshift and mean overdensity, return list of four Tinker
        parameters. If outside defined overdensity, return last valid number."""
        # Parameters at z=0
        lnDeltamean = np.log(Deltamean)
        A, a, b, c = [self.interp_A(lnDeltamean), self.interp_a(lnDeltamean), self.interp_b(lnDeltamean), self.interp_c(lnDeltamean)]
        # Redshift evolution
        logalpha = -(.75/np.log10(Deltamean/75.))**1.2
        alpha = 10**logalpha
        A *= (1+z)**-.14
        a *= (1+z)**-.06
        b *= (1+z)**-alpha
        return A, a, b, c
