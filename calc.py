from .utils import *

def derivative(func, x, h=None):
    """
    Calculate the derivative of a function at a given point using complex step differentiation.

    Parameters
    ----------
    func : callable
        The function for which the derivative is to be calculated.
    x : float
        The point at which the derivative is to be calculated.
    h : float, None
        The step size for the complex step. If None, it defaults to the machine epsilon for float64.

    Returns
    -------
    float
        The derivative of the function at the point x.
    """

    if h is None:
        h = np.finfo(np.float64).eps
    return np.imag(func(x + 1j * h)) / h