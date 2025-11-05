import numpy as np
import scipy.optimize as op
from scipy.interpolate import InterpolatedUnivariateSpline
from scipy.interpolate import RectBivariateSpline
from scipy import integrate
import cosmo


class ConcentrationConversion:

    def __init__(self, MCrelation, cosmology=None, setup_interp=False, interp_massdef=500):
        self.MCrelation = MCrelation
        if isinstance(MCrelation, float):
            self.MCrelation = float(MCrelation)
        elif MCrelation == 'Duffy08':
            pass
        elif MCrelation == 'Child18_obs':
            pass
        elif MCrelation == 'DK15':
            # We compute the c-M relation for an array of z_arr and
            # M_arr, and store the interpolation function in self
            z_arr = np.linspace(0, 2, 21)
            M_arr = np.logspace(13, 16, 301)
            rho_m = cosmology['Omega_m'] * cosmo.RHOCRIT
            Omh2 = cosmology['Omega_m']*cosmology['h']**2
            Obh2 = cosmology['Omega_b']*cosmology['h']**2
            fb = cosmology['Omega_b']/cosmology['Omega_m']
            k_arr = np.logspace(-4, 2, 400)

            # Eisenstein&Hu'99 transfer function (no wiggles)
            # EQ 6
            sound_horizon = 44.5 * np.log(9.83/Omh2) / np.sqrt(1 + 10*Obh2**.75)
            # EQ 31
            alphaGamma = 1 - .328 * np.log(431 * Omh2) * fb + .38 * np.log(22.3*Omh2) * fb**2
            # EQ 30
            Gamma = cosmology['Omega_m']*cosmology['h'] * (alphaGamma + (1-alphaGamma)/(1 + (.43*k_arr*cosmology['h']*sound_horizon)**4))
            # EQ 28
            q = k_arr * (2.7255/2.7)**2 / Gamma
            # EQ 29
            C0 = 14.2 + 731 / (1 + 62.5*q)
            L0 = np.log(2 * np.exp(1) + 1.8*q)
            TF = L0 / (L0 + C0 * q**2)
            # We only care about the derivative, not the normalization
            PK_EHsmooth = k_arr**cosmology['n_s'] * TF**2
            # Interpolation function for EQ 8, DK15
            n_of_k = InterpolatedUnivariateSpline(np.log(k_arr), np.log(PK_EHsmooth))

            # Normalized growth function
            integrand = lambda z_int: (1+z_int) / cosmo.Ez(z_int, cosmology)**3
            D_arr = np.array([cosmo.Ez(z, cosmology) * integrate.quad(integrand, z, 1e3)[0] for z in z_arr])
            D_arr /= D_arr[0]

            # Compute sigma(M, z=0)
            # Radius [M_arr]
            R = (3 * M_arr / (4 * np.pi * rho_m))**(1/3)
            R = np.append(R, 8)
            # [M_arr, k_arr]
            kR = k_arr[None, :] * R[:, None]
            # Window functions [M_arr, k_arr]
            window = 3 * (np.sin(kR)/kR**3 - np.cos(kR)/kR**2)
            # Integrand [M_arr, k_arr]
            integrand_sigma2 = PK_EHsmooth[None, :] * window[:, :]**2 * k_arr[None, :]**2
            # sigma^2 [z_arr, M_arr]
            sigma2 = .5/np.pi**2 * np.trapezoid(integrand_sigma2, k_arr, axis=-1)
            sigma = np.sqrt(sigma2[:-1]) * cosmology['sigma8']/np.sqrt(sigma2[-1])

            # EQ 12, DK15
            k_R = .69 * 2 * np.pi / R[:-1]
            n = n_of_k(np.log(k_R), nu=1)
            # EQ 4, DK15 [z_arr, M_arr]
            nu = 1.686 / sigma[None, :] / D_arr[:, None]
            # EQ 10, DK15 [M_arr]
            c_min = 6.58 + n*1.27
            nu_min = 7.28 + n*1.56
            # EQ 9, DK15 [z_arr, M_arr]
            c = .5*c_min * ((nu_min/nu)**1.08 + (nu/nu_min)**1.77)
            c[c > 30.] = 30.

            # Set up spline interpolation in z_arr and M_arr
            self.concentration = RectBivariateSpline(z_arr, M_arr, c)

        else:
            raise ValueError('Unknown mass-concentration relation:', MCrelation)

        if setup_interp:
            Min = np.logspace(13, 16, 8)
            z_arr = np.linspace(0, 2, 10)
            M200 = np.array([np.array([self.MDelta_to_M200(m, interp_massdef, z) for m in Min]) for z in z_arr])
            self.lnM_to_lnM200 = RectBivariateSpline(z_arr, np.log(Min), np.log(M200))

            # MDelta = np.array([np.array([self.M200_to_MDelta(m, interp_massdef, z) for m in Min]) for z in z_arr])
            # self.lnM200_to_lnM = RectBivariateSpline(z_arr, np.log(Min), np.log(MDelta))

    def calC200(self, m, z):
        """Return concentration c_200c. Input mass M200c [Msun/h]."""
        if self.MCrelation == 'Duffy08':
            m = np.atleast_1d(m)
            m[m < 1e9] = 1e9
            # return 6.71*(m/2.e12)**(-0.091)*(1.+z)**(-0.44)  # relaxed samples
            return 5.71*(m/2.e12)**(-0.084)*(1.+z)**(-0.47)  # full sample
        elif self.MCrelation == 'Child18_obs':
            # individual, all
            return 75.4 * (1+z)**-.422 * m**-.089
        elif self.MCrelation == 'DK15':
            c = self.concentration(z, m)
            # Reshape to match input...
            if c.shape == (1, 1):
                c = c[0][0]
            elif c.shape[0] == 1:
                c = c[0]
            return c
        else:
            return self.MCrelation

    # Actual input functions
    # Input in [Msun/h]
    def MDelta_to_M200(self, mc, overdensity, z):
        ratio = overdensity/200
        Mmin = mc * ratio / 4
        Mmax = mc * ratio * 4
        return op.brentq(self.mdiff_findM200, Mmin, Mmax, args=(mc, overdensity, z), xtol=1e-6)

    # Input in [Msun/h]
    def M200_to_MDelta(self, Minput, overdensity, z):
        ratio = 200/overdensity
        Mmin = Minput * ratio / 4
        Mmax = Minput * ratio * 4
        return op.brentq(self.mdiff_findMDelta, Mmin, Mmax, args=(Minput, overdensity, z), xtol=1e-6)

    # Functions used for conversion
    # calculate the coefficient for NFW aperture mass given c
    def calcoef(self, c):
        return np.log(1+c)-c/(1+c)

    # root function for concentration
    def diffc(self, c2, c200, ratio):
        return self.calcoef(c200)/self.calcoef(c2) - ratio*(c200/c2)**3

    def findc(self, c200, overdensity):
        ratio = 200./overdensity
        return op.brentq(self.diffc, .001, 40., args=(c200, ratio), xtol=1e-6)

    # Root function for mass
    def mdiff_findM200(self, m200, mc, overdensity, z):
        con = self.calC200(m200, z)
        con2 = self.findc(con, overdensity)
        return m200/mc - self.calcoef(con)/self.calcoef(con2)

    def mdiff_findMDelta(self, mguess, Minput, overdensity, z):
        conin = self.calC200(Minput, z)
        conguess = self.findc(conin, overdensity)
        return Minput/mguess - self.calcoef(conin)/self.calcoef(conguess)
