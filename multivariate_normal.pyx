import numpy as np
ln2pi = np.log(2.*np.pi)

def lnpdf(x, cov):
    """Natural logarithm of zero-mean multivariate normal probability for vector
    `x` and covariance matrix `cov`."""
    vals, vecs = np.linalg.eigh(cov)
    logdet     = np.sum(np.log(vals))
    U          = vecs / np.sqrt(vals)
    maha       = (np.matmul(x, U)**2.).sum(axis=-1)
    return -.5 * (maha + logdet + len(vals)*ln2pi)

def bivariate_normal(x, y, cov):
    """Return zero-mean bivariate normal distribution with covariance matrix
    `cov`. Input points are spanned by arrays `x` and `y` that may have
    different lengths."""
    cdef Py_ssize_t len_x = len(x)
    cdef Py_ssize_t len_y = len(y)
    cdef Py_ssize_t i, j, k
    cdef double[:] x_v = x
    cdef double[:] y_v = y
    chi2 = np.empty((len_x, len_y), dtype=float)
    cdef double[:,:] chi2_v = chi2
    cov_inv = np.linalg.inv(cov)
    cdef double[:,:] cov_inv_v = cov_inv
    cdef double[2] tmp

    for i in range(len_x):
      for j in range(len_y):
        for k in range(2):
          tmp[k] = cov_inv_v[k,0]*x_v[i] + cov_inv_v[k,1]*y_v[j]
        chi2_v[i,j] = x_v[i]*tmp[0] + y_v[j]*tmp[1]
    res = np.exp(-.5*chi2)/np.sqrt(np.linalg.det(cov))/(2*np.pi)
    return res


def trivariate_normal(x, y, z, cov):
    """Return zero-mean trivariate normal distribution with covariance matrix
    `cov`. Input points are spanned by arrays `x`, `y`, `z` that may have
    different lengths."""
    cdef Py_ssize_t len_x = len(x)
    cdef Py_ssize_t len_y = len(y)
    cdef Py_ssize_t len_z = len(z)
    cdef Py_ssize_t i, j, k, l
    cdef double[:] x_v = x
    cdef double[:] y_v = y
    cdef double[:] z_v = z
    chi2 = np.empty((len_x, len_y, len_z), dtype=float)
    cdef double[:,:,:] chi2_v = chi2
    cov_inv = np.linalg.inv(cov)
    cdef double[:,:] cov_inv_v = cov_inv
    cdef double[3] tmp

    for i in range(len_x):
      for j in range(len_y):
        for k in range(len_z):
          for l in range(3):
            tmp[l] = cov_inv_v[l,0]*x_v[i] + cov_inv_v[l,1]*y_v[j] + cov_inv_v[l,2]*z_v[k]
          chi2_v[i,j,k] = x_v[i]*tmp[0] + y_v[j]*tmp[1] + z_v[k]*tmp[2]
    res = np.exp(-.5*chi2)/np.sqrt(np.linalg.det(cov)*(2*np.pi)**3)
    return res

def bivariate_chi2_multivec(x, y, cov_inv):
    """Return chi2 for array of 2d vectors and an array of (inverse) covariance
    matrices. Everything must have the same length."""
    cdef Py_ssize_t i, j
    cdef Py_ssize_t len_vec = len(x)
    cdef double[:] x_v = x
    cdef double[:] y_v = y
    cdef double [:,:,:] cov_inv_v = cov_inv
    chi2 = np.empty(len_vec)
    cdef double[:] chi2_v = chi2
    cdef double[2] tmp

    for i in range(len_vec):
      for j in range(2):
        tmp[j] = cov_inv_v[i,j,0]*x_v[i] + cov_inv_v[i,j,1]*y_v[i]
      chi2_v[i] = x_v[i]*tmp[0] + y_v[i]*tmp[1]
    return chi2

def trivariate_chi2_multivec(x, y, z, cov_inv):
    """Return chi2 for array of 3d verctors and an array of (inverse) covariance
    matrices. Everything must have the same length."""
    cdef Py_ssize_t i, j
    cdef Py_ssize_t len_vec = len(x)
    cdef double[:] x_v = x 
    cdef double[:] y_v = y
    cdef double[:] z_v = z
    cdef double [:,:,:] cov_inv_v = cov_inv
    chi2 = np.empty(len_vec)
    cdef double[:] chi2_v = chi2
    cdef double[3] tmp

    for i in range(len_vec):
      for j in range(3):
        tmp[j] = cov_inv_v[i,j,0]*x_v[i] + cov_inv_v[i,j,1]*y_v[i] + cov_inv_v[i,j,2]*z_v[i]
      chi2_v[i] = x_v[i]*tmp[0] + y_v[i]*tmp[1] + z_v[i]*tmp[2]
    return chi2

