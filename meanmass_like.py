import numpy as np


ln2pi = np.log(2.*np.pi)


def lnlike(data, model, cov):
    """Return the ln-likelihood."""
    diff = model[model > 0] - data
    lnlike = -.5 * (np.dot(diff, np.linalg.solve(cov, diff))
                    + len(data)*ln2pi
                    + np.linalg.slogdet(cov)[1])
    return lnlike
