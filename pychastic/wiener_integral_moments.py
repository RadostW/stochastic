import numpy as np

poly = np.polynomial.Polynomial


def E(alpha) -> poly:
    """
    Calculate expected value of mixed winer integral E(I_alpha).
    Entries in multindex alpha coresspond to:

    * 0 - time dimension    
    * positive integer - wiener dimension

    Parameters
    ----------
    alpha : array of integers

    Returns
    -------
    numpy.polynomial.Polynomial
        polynomial in time

    """
    if any(a != 0 for a in alpha):
        return poly([0])
    else:
        return poly([1]).integ(len(alpha))


def E2(alpha, beta=None) -> poly:
    """
    Calculate expected value of a product of two mixed winer integrals E(I_alpha*I_beta) or
    E(I_alpha^2) if beta=None.
    Entries in multindices alpha and beta coresspond to:
    
    * 0 - time dimension    
    * positive integer - wiener dimension

    Parameters
    ----------
    alpha : array of integers
        multindex of the first integral
    beta : array of integers or None
        multindex of the second integral

    Returns
    -------
    numpy.polynomial.Polynomial
        polynomial in time

    """
    beta = alpha if beta is None else beta

    if len(alpha) == 0:
        return E(beta)
    if len(beta) == 0:
        return E(alpha)

    # recursion
    w = poly([0])
    if alpha[-1] == beta[-1] != 0:
        w += E2(alpha[:-1], beta[:-1]).integ()
    if alpha[-1] == 0:
        w += E2(alpha[:-1], beta).integ()
    if beta[-1] == 0:
        w += E2(alpha, beta[:-1]).integ()
    return w
