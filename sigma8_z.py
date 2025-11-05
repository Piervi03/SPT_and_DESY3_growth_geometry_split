import numpy as np


def rescale_Pk(z, k, Pk, zlim, zmid, sigma8_z, rescale):
    """Rescale `Pk` within each `zlim` interval to `sigma8_z` at `zmid`."""
    # Compute sigma_8^2(zmid)
    kR = 8 * k
    window = 3 * (np.sin(kR)/kR**3 - np.cos(kR)/kR**2)
    if rescale:
        zmid_idx = np.digitize(zmid, z) - 1
        integrand_sigma2 = Pk[zmid_idx, :] * window[None, :]**2 * k[None, :]**3
        sigma2 = .5/np.pi**2 * np.trapezoid(integrand_sigma2, np.log(k), axis=1)
        # Rescale matter power spectrum to sigma8_z
        zlim_idx = np.digitize(z, zlim) - 1
        Pk *= (sigma8_z[zlim_idx]**2/sigma2[zlim_idx])[:, None]
    # Redshift array for plotting
    z_out = zmid.copy()
    for i in range(len(zmid)):
        z_out = np.append(z_out, np.linspace(zlim[i], zlim[i+1]-.01, 4))
    z_out = sorted(z_out)
    # Compute sigma_8(z_out)
    z_idx = np.digitize(z_out, z) - 1
    integrand_sigma2 = Pk[z_idx, :] * window[None, :]**2 * k[None, :]**3
    sigma8_zout = np.sqrt(.5 * np.trapezoid(integrand_sigma2, np.log(k), axis=1))/np.pi
    return Pk, z_out, sigma8_zout
