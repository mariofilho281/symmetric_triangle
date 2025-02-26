import numpy as np
from numpy.polynomial import Polynomial
from scipy.optimize import root
from polynomials import f as F, g as G, Dfa, Dfc, Dga, Dgc, bfun, dfun, efun, \
    ffun, sfun, tfun, e3fun


# ----------------------------------------------------------- General utilities
def valid_distribution(e1, e2, e3):
    """
    Checks if the given correlators `e1`, `e2`, and `e3` satisfy the positivity
    constraints required for a valid probability distribution.

    The function evaluates a set of inequalities derived from the requirement that
    the probabilities of all measurement outcomes must be non-negative. These
    inequalities define the valid region for the correlators in the parameter space.

    Parameters:
    -----------
    e1 : float or numpy.ndarray
        The one-body correlator. Can be a single float or a numpy array of floats.
    e2 : float or numpy.ndarray
        The two-body correlator. Can be a single float or a numpy array of floats.
    e3 : float or numpy.ndarray
        The three-body correlator. Can be a single float or a numpy array of floats.

    Returns:
    --------
    bool or numpy.ndarray
        - If the inputs are floats, returns `True` if the correlators satisfy the
          positivity constraints, and `False` otherwise.
        - If the inputs are numpy arrays, returns a boolean array of the same shape,
          where each element indicates whether the corresponding set of correlators
          satisfies the constraints.

    Notes:
    ------
    - The constraints are derived from the requirement that the probabilities of all
      measurement outcomes must be non-negative.
    - If `e1`, `e2`, and `e3` are numpy arrays, they must have the same shape.

    Examples:
    ---------
    >>> # Example with floats
    >>> e1, e2, e3 = 0.5, 0.5, 0.25
    >>> is_valid = valid_distribution(e1, e2, e3)
    >>> print("Is the distribution valid?", is_valid)

    >>> # Example with numpy arrays
    >>> e1 = np.array([0.5, 0.6])
    >>> e2 = np.array([0.5, 0.4])
    >>> e3 = np.array([0.25, 0.3])
    >>> is_valid = valid_distribution(e1, e2, e3)
    >>> print("Are the distributions valid?", is_valid)
    """
    valid = e2 >= 2*e1 - 1
    valid &= e2 >= -2*e1 - 1
    valid &= e3 >= -1 + e1 + e2
    valid &= e3 <= 1 + e1 - e2
    valid &= e3 >= -1 - 3*e1 - 3*e2
    valid &= e3 <= 1 - 3*e1 + 3*e2
    return valid


def bit_flip(e1, e2, e3):
    """
    Applies a bit-flip transformation to the correlators `e1`, `e2`, and `e3`.

    The bit-flip transformation negates `e1` and `e3` while leaving `e2` unchanged.

    Parameters:
    -----------
    e1 : float or numpy.ndarray
        The one-body correlator. Can be a single float or a numpy array of floats.
    e2 : float or numpy.ndarray
        The two-body correlator. Can be a single float or a numpy array of floats.
    e3 : float or numpy.ndarray
        The three-body correlator. Can be a single float or a numpy array of floats.

    Returns:
    --------
    tuple
        A tuple containing the transformed correlators `(-e1, e2, -e3)`. If the inputs
        are numpy arrays, the transformation is applied element-wise.

    Examples:
    ---------
    >>> # Example with floats
    >>> e1, e2, e3 = 0.5, 0.5, 0.25
    >>> transformed = bit_flip(e1, e2, e3)
    >>> print("Transformed correlators:", transformed)

    >>> # Example with numpy arrays
    >>> e1 = np.array([0.5, 0.6])
    >>> e2 = np.array([0.5, 0.4])
    >>> e3 = np.array([0.25, 0.3])
    >>> transformed = bit_flip(e1, e2, e3)
    >>> print("Transformed correlators:", transformed)
    """
    return -e1, e2, -e3


def fun(x, point):
    """
    Defines a system of equations to solve for the parameters `a` and `c` of the cyan surface,
    given the correlators `e1` and `e2`.

    This function evaluates two equations, `F(a, c, e1, e2)` and `G(a, c, e1, e2)`, which
    define the equations on the parameters `a` and `c` for points lying on the cyan surface.
    The system is designed to be solved using numerical solvers like `scipy.optimize.root`.

    Parameters:
    -----------
    x : tuple or numpy.ndarray
        A tuple or array containing the parameters `a` and `c` of the cyan surface.
        These are the variables to be solved for.
    point : tuple or numpy.ndarray
        A tuple or array containing the correlators `e1` and `e2`. These are fixed
        values used in the evaluation of the equations.

    Returns:
    --------
    tuple
        A tuple containing the evaluated values of the equations `F(a, c, e1, e2)`
        and `G(a, c, e1, e2)`. These values represent the residuals of the equations,
        which should be zero if evaluated in a solution.

    Notes:
    ------
    - The functions `F` and `G` are defined in the polynomials.py file. They
      represent the equations that define the cyan surface in terms of the
      parameters `a` and `c` and the correlators `e1` and `e2`.
    - This function is intended to be used with numerical solvers like `scipy.optimize.root`
      to find values of `a` and `c` that satisfy both equations simultaneously.
    - In general, this system has multiple solutions, so one has to check if the
      positivity constraints are satisfied.

    Examples:
    ---------
    >>> from scipy.optimize import root
    >>> from triangle_inequalities import fun, e3fun
    >>> e1, e2 = 1/3, -5/27
    >>> a, c = root(fun, (0.25, 0.25), args=((e1, e2), )).x
    >>> print(f'a = {a}, c = {c}')
    >>> print(f'E3 = {e3fun(a, c, e1, e2)}')
    """
    return F(*x, *point), G(*x, *point)


def jac(x, point):
    """
    Computes the Jacobian matrix of the system of equations defined in `fun`.

    The Jacobian matrix is a 2x2 matrix containing the partial derivatives of the
    equations `F(a, c, e1, e2)` and `G(a, c, e1, e2)` with respect to the variables
    `a` and `c`. This matrix is used to expedite numerical solvers like
    `scipy.optimize.root` by providing information about the local behavior of the
    system of equations.

    Parameters:
    -----------
    x : tuple or numpy.ndarray
        A tuple or array containing the parameters `a` and `c` of the cyan surface.
        These are the variables with respect to which the partial derivatives are
        computed.
    point : tuple or numpy.ndarray
        A tuple or array containing the correlators `e1` and `e2`. These are fixed
        values used in the evaluation of the partial derivatives.

    Returns:
    --------
    numpy.ndarray
        A 2x2 Jacobian matrix of the form:
        \[
        \begin{bmatrix}
        \frac{\partial F}{\partial a} & \frac{\partial F}{\partial c} \\
        \frac{\partial G}{\partial a} & \frac{\partial G}{\partial c}
        \end{bmatrix}
        \]
        where the partial derivatives are evaluated at `(a, c, e1, e2)`.

    Notes:
    ------
    - The functions `Dfa`, `Dfc`, `Dga`, and `Dgc` are defined in the polynomials.py
      file. These functions compute the partial derivatives of `F` and `G` with
      respect to `a` and `c`.
    - Providing the Jacobian matrix can significantly improve the performance and
      accuracy of numerical solvers like `scipy.optimize.root`.

    Examples:
    ---------
    >>> from scipy.optimize import root
    >>> from triangle_inequalities import fun, jac, e3fun
    >>> e1, e2 = 1/3, -5/27
    >>> a, c = root(fun, (0.25, 0.25), args=((e1, e2), ), jac=jac).x
    >>> print(f'a = {a}, c = {c}')
    >>> print(f'E3 = {e3fun(a, c, e1, e2)}')
    """
    return np.array([[Dfa(*x, *point), Dfc(*x, *point)], 
                     [Dga(*x, *point), Dgc(*x, *point)]])


# -------------------------------------------------------- Network inequalities
def GHZ(e1, e2, e3, flip=False):
    """
    Computes the inequality for the GHZ surface involving the correlators `e1`, `e2`,
    and `e3`. If `GHZ >= 0`, the behavior is not in the GHZ region. Optionally, the
    behavior can be "flipped" using the `bit_flip` function before evaluation.

    Parameters:
    -----------
    e1 : float or numpy.ndarray
        The one-body correlator.
    e2 : float or numpy.ndarray
        The two-body correlator.
    e3 : float or numpy.ndarray
        The three-body correlator.
    flip : bool, optional
        If `True`, the correlators `e1`, `e2`, and `e3` are flipped using the `bit_flip`
        function before evaluating `GHZ` (default is `False`).

    Returns:
    --------
    float or numpy.ndarray
        The value of the inequality `GHZ`. If `GHZ >= 0`, the behavior is not in
        the GHZ region; otherwise, the behavior might be nonlocal, in case it also
        violates the flipped version of the inequality.

    Examples:
    ---------
    >>> from triangle_inequalities import GHZ
    >>> e1, e2, e3 = 0, 1, 0
    >>> violated = [GHZ(e1, e2, e3, flip=flip) < 0 for flip in (False, True)]
    >>> non_local = all(violated)
    >>> print(f'Is GHZ nonlocal? Answer: {non_local}.')
    """
    if flip:
        e1, e2, e3 = bit_flip(e1, e2, e3)
    GHZ = 8*(1+e1)**3*(11 + 37*e1 + 28*e1**2 - 4*e1**3 + 8*e2 + 13*e1*e2 
                       + e2**2 - 11*e3 - 18*e1*e3 - 3*e2*e3 + 2*e3**2) \
        - (3 + 5*e1 + e2 - e3)**4 + np.sqrt((3 + 5*e1 + e2 - e3)**2 
                                            - 8*(1+e1)**3)\
            *((3 + 5*e1 + e2 - e3)**3 - 4*(1+e1)**3*(7 + 11*e1 + e2 - 3*e3))
    return np.where((3 + 5*e1 + e2 - e3)**2 >= 8*(1 + e1)**3, GHZ, np.inf)


def W1(e1, e2, e3, flip=False):
    """
    Computes the inequality for the W green surface involving the correlators `e1`,
    `e2`, and `e3`. If `W1 >= 0`, the behavior is not in the W region. Optionally,
    the behavior can be "flipped" using the `bit_flip` function before evaluation.

    Parameters:
    -----------
    e1 : float or numpy.ndarray
        The one-body correlator.
    e2 : float or numpy.ndarray
        The two-body correlator.
    e3 : float or numpy.ndarray
        The three-body correlator.
    flip : bool, optional
        If `True`, the correlators `e1`, `e2`, and `e3` are flipped using the `bit_flip`
        function before evaluating `W1` (default is `False`).

    Returns:
    --------
    float or numpy.ndarray
        The value of the inequality `W1`. If `W1 >= 0`, the behavior is not in the
        W region; otherwise, the behavior might be nonlocal, in case it also
        violates the other W inequalities.

    Examples:
    ---------
    >>> from triangle_inequalities import W1, W2, W3, W4, W5
    >>> e1, e2, e3 = 1/3, -1/3, -1
    >>> violated = [ineq(e1, e2, e3)<0 for ineq in (W1, W2, W3, W4, W5)]
    >>> non_local = all(violated)
    >>> print(f'Is W nonlocal? Answer: {non_local}.')
    """
    if flip:
        e1, e2, e3 = bit_flip(e1, e2, e3)
    W1 = -4*e1**2 + 4*e1**3 + (1 + e2 + np.sqrt((1+e2)**2 - 4*e1**2))\
        *(1 - e1 + e2 + e3 - 2*e1*e2)
    return W1


def W2(e1, e2, e3, flip=False):
    """
    Computes the inequality for the W purple surface involving the correlators `e1`,
    `e2`, and `e3`. If `W2 >= 0`, the behavior is not in the W region. Optionally,
    the behavior can be "flipped" using the `bit_flip` function before evaluation.

    Parameters:
    -----------
    e1 : float or numpy.ndarray
        The one-body correlator.
    e2 : float or numpy.ndarray
        The two-body correlator.
    e3 : float or numpy.ndarray
        The three-body correlator.
    flip : bool, optional
        If `True`, the correlators `e1`, `e2`, and `e3` are flipped using the `bit_flip`
        function before evaluating `W2` (default is `False`).

    Returns:
    --------
    float or numpy.ndarray
        The value of the inequality `W2`. If `W2 >= 0`, the behavior is not in the
        W region; otherwise, the behavior might be nonlocal, in case it also
        violates the other W inequalities.

    Examples:
    ---------
    >>> from triangle_inequalities import W1, W2, W3, W4, W5
    >>> e1, e2, e3 = 1/3, -1/3, -1
    >>> violated = [ineq(e1, e2, e3)<0 for ineq in (W1, W2, W3, W4, W5)]
    >>> non_local = all(violated)
    >>> print(f'Is W nonlocal? Answer: {non_local}.')
    """
    if flip:
        e1, e2, e3 = bit_flip(e1, e2, e3)
    # W2 = e3 - (68*e1**6 + 11*e1**5*e2 - 411*e1**5 - 4*e1**4*e2**2 - 123*e1**4*e2 + 795*e1**4 + 141*e1**3*e2**2 + 495*e1**3*e2 - 784*e1**3 - 557*e1**2*e2**2 - 407*e1**2*e2 + 364*e1**2 + 543*e1*e2**2 + 18*e1*e2 - 13*e1 - 99*e2**2 - 34*e2 + np.sqrt(2)*np.sqrt(1 - e1)*(-8*e1**6 - 2*e1**5*e2 + 140*e1**5 + 24*e1**4*e2 - 378*e1**4 - 25*e1**3*e2**2 - 203*e1**3*e2 + 430*e1**3 + 228*e1**2*e2**2 + 290*e1**2*e2 - 252*e1**2 - 349*e1*e2**2 - e1*e2 + 10*e1 + 70*e2**2 + 24*e2 + 2) - 3)/(-4*e1**5 - e1**4*e2 + 49*e1**4 + 33*e1**3*e2 - 109*e1**3 - 131*e1**2*e2 + 111*e1**2 + 131*e1*e2 - 51*e1 - 24*e2 + np.sqrt(2)*np.sqrt(1 - e1)*(-14*e1**4 - 6*e1**3*e2 + 52*e1**3 + 53*e1**2*e2 - 59*e1**2 - 84*e1*e2 + 38*e1 + 17*e2 + 3) - 4)
    W2 = 8*e1**5 - 15*e1**4 - 16*e1**3*e2 + 6*e1**3*e3 + 22*e1**3 + 16*e1**2*e2**2 - 8*e1**2*e2*e3 - 2*e1**2*e2 + e1**2*e3**2 - 12*e1**2*e3 - 20*e1**2 - 6*e1*e2**2 + 10*e1*e2*e3 + 12*e1*e2 - 2*e1*e3**2 + 10*e1*e3 + 10*e1 - e2**2 - 8*e2*e3 - 6*e2 + 2*e3**2 - 1
    return np.where(e1 >= 1/3, W2, -np.inf)


def W3(e1, e2, e3, flip=False):
    """
    Computes the inequality for the W red surface involving the correlators `e1`,
    `e2`, and `e3`. If `W3 >= 0`, the behavior is not in the W region. Optionally,
    the behavior can be "flipped" using the `bit_flip` function before evaluation.

    Parameters:
    -----------
    e1 : float or numpy.ndarray
        The one-body correlator.
    e2 : float or numpy.ndarray
        The two-body correlator.
    e3 : float or numpy.ndarray
        The three-body correlator.
    flip : bool, optional
        If `True`, the correlators `e1`, `e2`, and `e3` are flipped using the `bit_flip`
        function before evaluating `W3` (default is `False`).

    Returns:
    --------
    float or numpy.ndarray
        The value of the inequality `W3`. If `W3 >= 0`, the behavior is not in the
        W region; otherwise, the behavior might be nonlocal, in case it also
        violates the other W inequalities.

    Examples:
    ---------
    >>> from triangle_inequalities import W1, W2, W3, W4, W5
    >>> e1, e2, e3 = 1/3, -1/3, -1
    >>> violated = [ineq(e1, e2, e3)<0 for ineq in (W1, W2, W3, W4, W5)]
    >>> non_local = all(violated)
    >>> print(f'Is W nonlocal? Answer: {non_local}.')
    """
    if flip:
        e1, e2, e3 = bit_flip(e1, e2, e3)
    valid = ((2048*e1**12 - 1024*e1**11*e2 - 3832*e1**11 + 2196*e1**10*e2 + 5508*e1**10 + 1062*e1**9*e2**2 + 2796*e1**9*e2 + 12198*e1**9 - 729*e1**8*e2**3 - 543*e1**8*e2**2 - 23427*e1**8*e2 - 20061*e1**8 - 1404*e1**7*e2**3 + 12636*e1**7*e2**2 + 29532*e1**7*e2 + 8084*e1**7 + 1368*e1**6*e2**3 - 2112*e1**6*e2**2 - 27680*e1**6*e2 - 1416*e1**6 - 2484*e1**5*e2**4 - 156*e1**5*e2**3 + 43568*e1**5*e2**2 + 2324*e1**5*e2 + 9212*e1**5 - 3456*e1**4*e2**4 - 18162*e1**4*e2**3 + 19706*e1**4*e2**2 - 25430*e1**4*e2 - 17250*e1**4 - 2664*e1**3*e2**4 - 17620*e1**3*e2**3 + 33876*e1**3*e2**2 + 44052*e1**3*e2 + 9340*e1**3 + 432*e1**2*e2**5 + 848*e1**2*e2**4 - 20128*e1**2*e2**3 - 31704*e1**2*e2**2 - 12700*e1**2*e2 - 500*e1**2 + 2268*e1*e2**4 + 5100*e1*e2**3 + 2682*e1*e2**2 - 624*e1*e2 - 506*e1 + 243*e2**3 + 477*e2**2 + 369*e2 + 71 <= 0)
             & (4*e1**4 - 8*e1**3 + 4*e1**2*e2 + 12*e1**2 + 2*e1*e2**2 + 4*e1*e2
                + 2*e1 - e2**2 - 2*e2 - 1 >= 0)
             & (e3 >= 6 - 3*np.sqrt(5)))
    W3 = np.where(valid, 
                  216*e1**5 - 108*e1**4*e2 - 162*e1**4 - 54*e1**3*e2 + 270*e1**3*e3 - 54*e1**3 + 162*e1**2*e2**2 - 216*e1**2*e2*e3 + 351*e1**2*e2 + 54*e1**2*e3**2 - 270*e1**2*e3 + 189*e1**2 - 54*e1*e2**3 + 54*e1*e2**2*e3 - 216*e1*e2**2 - 18*e1*e2*e3**2 + 162*e1*e2*e3 - 270*e1*e2 + 2*e1*e3**3 + 18*e1*e3**2 + 108*e1*e3 - 108*e1 + 27*e2**3 + 81*e2**2 - 9*e2*e3**2 + 81*e2 + 2*e3**3 - 9*e3**2 + 27, 
                  -np.inf)
    return W3


def W4(e1, e2, e3, flip=False):
    """
    Computes the inequality for the W yellow surface involving the correlators `e1`,
    `e2`, and `e3`. If `W4 >= 0`, the behavior is not in the W region. Optionally,
    the behavior can be "flipped" using the `bit_flip` function before evaluation.

    Parameters:
    -----------
    e1 : float or numpy.ndarray
        The one-body correlator.
    e2 : float or numpy.ndarray
        The two-body correlator.
    e3 : float or numpy.ndarray
        The three-body correlator.
    flip : bool, optional
        If `True`, the correlators `e1`, `e2`, and `e3` are flipped using the `bit_flip` 
        function before evaluating `W4` (default is `False`).

    Returns:
    --------
    float or numpy.ndarray
        The value of the inequality `W4`. If `W4 >= 0`, the behavior is not in the
        W region; otherwise, the behavior might be nonlocal, in case it also
        violates the other W inequalities.

    Examples:
    ---------
    >>> from triangle_inequalities import W1, W2, W3, W4, W5
    >>> e1, e2, e3 = 1/3, -1/3, -1
    >>> violated = [ineq(e1, e2, e3)<0 for ineq in (W1, W2, W3, W4, W5)]
    >>> non_local = all(violated)
    >>> print(f'Is W nonlocal? Answer: {non_local}.')
    """
    if flip:
        e1, e2, e3 = bit_flip(e1, e2, e3)
    W4 = 256*e1**8 - 256*e1**7*e2 - 882*e1**7 + 64*e1**6*e2**2 + 504*e1**6*e2 + 88*e1**6*e3 + 1013*e1**6 + 212*e1**5*e2**2 - 232*e1**5*e2*e3 + 161*e1**5*e2 + 36*e1**5*e3**2 - 299*e1**5*e3 - 252*e1**5 - 184*e1**4*e2**3 + 168*e1**4*e2**2*e3 - 766*e1**4*e2**2 - 24*e1**4*e2*e3**2 + 476*e1**4*e2*e3 - 919*e1**4*e2 - 8*e1**4*e3**3 - 62*e1**4*e3**2 + 409*e1**4*e3 - 345*e1**4 + 30*e1**3*e2**4 - 40*e1**3*e2**3*e3 + 258*e1**3*e2**3 + 4*e1**3*e2**2*e3**2 - 138*e1**3*e2**2*e3 + 432*e1**3*e2**2 + 8*e1**3*e2*e3**3 - 66*e1**3*e2*e3**2 - 192*e1**3*e2*e3 + 530*e1**3*e2 - 2*e1**3*e3**4 + 42*e1**3*e3**3 + 16*e1**3*e3**2 - 278*e1**3*e3 + 270*e1**3 - 7*e1**2*e2**4 - 36*e1**2*e2**3*e3 + 118*e1**2*e2**3 + 86*e1**2*e2**2*e3**2 - 230*e1**2*e2**2*e3 + 364*e1**2*e2**2 - 52*e1**2*e2*e3**3 + 154*e1**2*e2*e3**2 - 152*e1**2*e2*e3 + 98*e1**2*e2 + 9*e1**2*e3**4 - 42*e1**2*e3**3 + 12*e1**2*e3**2 + 82*e1**2*e3 - 61*e1**2 - 3*e1*e2**5 + 13*e1*e2**4*e3 - 52*e1*e2**4 - 22*e1*e2**3*e3**2 + 112*e1*e2**3*e3 - 290*e1*e2**3 + 18*e1*e2**2*e3**3 - 88*e1*e2**2*e3**2 + 250*e1*e2**2*e3 - 404*e1*e2**2 - 7*e1*e2*e3**4 + 32*e1*e2*e3**3 - 62*e1*e2*e3**2 + 136*e1*e2*e3 - 147*e1*e2 + e1*e3**5 - 4*e1*e3**4 + 6*e1*e3**3 - 4*e1*e3**2 + e1*e3 + e2**5 - 3*e2**4*e3 + 29*e2**4 + 2*e2**3*e3**2 - 36*e2**3*e3 + 98*e2**3 + 2*e2**2*e3**3 - 2*e2**2*e3**2 - 50*e2**2*e3 + 98*e2**2 - 3*e2*e3**4 + 12*e2*e3**3 - 2*e2*e3**2 - 36*e2*e3 + 29*e2 + e3**5 - 3*e3**4 + 2*e3**3 + 2*e3**2 - 3*e3 + 1
    return W4


def W5(e1, e2, e3, flip=False):
    """
    Computes the inequality for the W cyan surface involving the correlators `e1`,
    `e2`, and `e3`. If `W5 >= 0`, the behavior is not in the W region. Optionally,
    the behavior can be "flipped" using the `bit_flip` function before evaluation.

    Parameters:
    -----------
    e1 : float or numpy.ndarray
        The one-body correlator.
    e2 : float or numpy.ndarray
        The two-body correlator.
    e3 : float or numpy.ndarray
        The three-body correlator.
    flip : bool, optional
        If `True`, the correlators `e1`, `e2`, and `e3` are flipped using the `bit_flip`
        function before evaluating `W5` (default is `False`).

    Returns:
    --------
    float or numpy.ndarray
        The value of the inequality `W5`. If `W5 >= 0`, the behavior is not in the
        W region; otherwise, the behavior might be nonlocal, in case it also
        violates the other W inequalities.

    Examples:
    ---------
    >>> from triangle_inequalities import W1, W2, W3, W4, W5
    >>> e1, e2, e3 = 1/3, -1/3, -1
    >>> violated = [ineq(e1, e2, e3)<0 for ineq in (W1, W2, W3, W4, W5)]
    >>> non_local = all(violated)
    >>> print(f'Is W nonlocal? Answer: {non_local}.')
    """
    if flip:
        e1, e2, e3 = bit_flip(e1, e2, e3)
    
    e1, e2 = np.broadcast_arrays(*np.atleast_1d(e1, e2))
    e3_W5 = np.full_like(e1.flatten(), np.inf)
    initial_guesses = np.array([[0.28, 0.44], 
                                [0.36, 0.46], 
                                [0.32, 0.3], 
                                [0.3, 0.5], 
                                [0.4, 0.44]])
    
    # intersection with D0 W Wflip plane
    e = np.linspace(1/2, 1/3, 40)
    border1 = np.vstack((e, (2*np.sqrt(2*(3*e-1))*(1-3*e) + 36*e - 17)/27)).T
    params1 = np.vstack(((2 + np.sqrt(2*(3*e-1)))/6, (1 - np.sqrt(2*(3*e-1)))/3)).T
    
    # intersection with red surface
    e = np.linspace(1/3, 1, 40)
    a = np.empty_like(e)
    for i, e_ in enumerate(e):
        a[i] = Polynomial([4*e_**2*(1 + e_)**2, 
                           -5*e_ - 16*e_**2 - 50*e_**3 - 24*e_**4 - e_**5, 
                           1 + 12*e_ + 34*e_**2 + 84*e_**3 + 45*e_**4, 
                           -12*e_ - 24*e_**2 - 60*e_**3, 
                           16*e_**2]).roots()[0].real
    border2 = np.vstack(((1-3*a)*e/(a-e), 
                         (-a + 3*e - 6*a*e + 4*a**3*e - 4*a**2*e**2)/(a - e))).T
    params2 = np.vstack((a, a)).T
    border = np.vstack((border1, border2))
    params = np.vstack((params1, params2))
    
    for i, (e1_, e2_) in enumerate(zip(e1.flatten(), e2.flatten())):
        
        if (e1_ > 1/2 or e1_ < np.sqrt(5)-2 
                or e2_ > 9-4*np.sqrt(5) or e2_ < -5/27 
                or e2_ < (-5 + 2*(3*e1_-1)*(6 - np.sqrt(2*(3*e1_-1))))/27):
            continue
        
        distances_square = np.sum((border - np.array([e1_, e2_]))**2, axis=1)
        initial_point = border[np.argmin(distances_square)]
        initial_params = params[np.argmin(distances_square)]
        displacement = np.array([e1_, e2_]) - initial_point
        distance = np.sqrt(np.sum(displacement**2))
        direction = displacement/distance
        step = 0.01
        alphas = np.arange(0, distance, step)
        x0 = initial_params
        
        for alpha in alphas:
            point = initial_point + alpha*direction
            solution = root(fun, x0, args=point, jac=jac)
            x0 = solution.x
        
        solution = root(fun, x0, args=((e1_, e2_), ), jac=jac)
        a, c = solution.x
        b = bfun(a, c, e1_, e2_)
        d = dfun(a, c, e1_, e2_)
        e = efun(a, c, e1_, e2_)
        f = ffun(a, c, e1_, e2_)
        s = sfun(a, c, e1_, e2_)
        t = tfun(a, c, e1_, e2_)
        e3_ = e3fun(a, c, e1_, e2_)
        if (a>=0 and b>=0 and a+b<=1 and c>=0 and d>=0 and c+d<=1 and e>=0 
                and f>=0 and e+f<=1 and s>=0 and s<=1 
                and np.isclose(t, 1, rtol=1e-3) and e3_ < e3_W5[i]):
            e3_W5[i] = e3fun(a, c, e1_, e2_)
        
        for x0 in initial_guesses:
            solution = root(fun, x0, args=((e1_, e2_), ), method='hybr', jac=jac)
            a, c = solution.x
            b = bfun(a, c, e1_, e2_)
            d = dfun(a, c, e1_, e2_)
            e = efun(a, c, e1_, e2_)
            f = ffun(a, c, e1_, e2_)
            s = sfun(a, c, e1_, e2_)
            t = tfun(a, c, e1_, e2_)
            e3_ = e3fun(a, c, e1_, e2_)
            if (a>=0 and b>=0 and a+b<=1 and c>=0 and d>=0 and c+d<=1 and e>=0 
                    and f>=0 and e+f<=1 and s>=0 and s<=1 
                    and np.isclose(t, 1, rtol=1e-3) and e3_ < e3_W5[i]):
                e3_W5[i] = e3fun(a, c, e1_, e2_)

    W5 = e3 - e3_W5
    return W5.reshape(e1.shape)


# ------------------------------ Generate points on boundary (given projection)
def GHZ_point(e1, e3):
    """
    Computes the values of `e2` for points on the GHZ surface, given the
    correlators `e1` and `e3`.

    The function evaluates a mathematical expression to compute `e2` for points lying
    on the GHZ surface. It also ensures that the computed `e2` values correspond
    to valid probability distributions by checking the `valid_distribution` function.
    If the distribution is invalid, the function returns `np.inf` for those points.

    Parameters:
    -----------
    e1 : float or numpy.ndarray
        The one-body correlator. Can be a single float or a numpy array of floats.
    e3 : float or numpy.ndarray
        The three-body correlator. Can be a single float or a numpy array of floats.

    Returns:
    --------
    numpy.ndarray
        An array of `e2` values corresponding to points on the GHZ surface.
        If the computed `e2` values do not correspond to a valid probability
        distribution, the function returns `np.inf` for those points.

    Notes:
    ------
    - The `valid_distribution` function is used to check if the computed `e2` values
      correspond to valid probability distributions.
    - This function is not guaranteed to work for points on the surface boundary.

    Examples:
    ---------
    >>> # Example with floats
    >>> e1, e3 = 0, 0
    >>> e3 = GHZ_point(e1, e3)
    >>> print(f"Point on green surface: {e1, e2, e3}")

    >>> # Example with numpy arrays
    >>> e1 = np.linspace(-1, 1, 30)
    >>> e3 = np.linspace(-1, 1, 30)
    >>> e1, e3 = np.meshgrid(e1, e3)
    >>> e2 = GHZ_point(e1, e3)
    >>> e1, e2, e3 = e1.flatten(), e2.flatten(), e3.flatten()
    >>> GHZ_surface = np.vstack((e1, e2, e3)).T
    """
    e1, e3 = np.atleast_1d(np.array(e1), np.array(e3))
    e2 = np.full_like(e1.flatten(), np.inf, float)
    for i, (e1_, e3_) in enumerate(zip(e1.flatten(), e3.flatten())):
        roots = np.roots([4, 0, 0, -8 - 12 * e1_ + 4 * e3_, 3 + 6 * e1_ + 3 * e1_ ** 2])
        b = roots[~np.iscomplex(roots)].real
        a = (1 - 2 * b ** 2 + e1_) / (4 * b)
        valid = (a >= 0) & (b >= 0) & (a + b <= 1)
        if valid.sum() >= 1:
            a, b = a[valid][0], b[valid][0]
            print(f'{a=}, {b=}, {e1_=}, {e3_=}, {e2[i]=}')
            e2[i] = 1 - 8*a*b + 4*a**2*b - 4*b**2 + 12*a*b**2 + 4*b**3
            print(f'{a=}, {b=}, {e1_=}, {e3_=}, {e2[i]=}')

    return e2.reshape(e1.shape)
def W1_point(e1, e2):
    """
    Computes the values of `e3` for points on the W green surface, given the correlators
    `e1` and `e2`.

    The function evaluates a mathematical expression to compute `e3` for points lying
    on the W green surface. It also ensures that the computed `e3` values correspond
    to valid probability distributions by checking the `valid_distribution` function.
    If the distribution is invalid, the function returns `np.inf` for those points.

    Parameters:
    -----------
    e1 : float or numpy.ndarray
        The one-body correlator. Can be a single float or a numpy array of floats.
    e2 : float or numpy.ndarray
        The two-body correlator. Can be a single float or a numpy array of floats.

    Returns:
    --------
    numpy.ndarray
        An array of `e3` values corresponding to points on the W green surface.
        If the computed `e3` values do not correspond to a valid probability
        distribution, the function returns `np.inf` for those points.

    Notes:
    ------
    - The `valid_distribution` function is used to check if the computed `e3` values
      correspond to valid probability distributions.
    - This function is not guaranteed to work for points on the surface boundary.

    Examples:
    ---------
    >>> # Example with floats
    >>> e1, e2 = 0, 0
    >>> e3 = W1_point(e1, e2)
    >>> print(f"Point on green surface: {e1, e2, e3}")

    >>> # Example with numpy arrays
    >>> e1 = np.linspace(-1, 1, 30)
    >>> e2 = np.linspace(-1/3, 1, 20)
    >>> e1, e2 = np.meshgrid(e1, e2)
    >>> e3 = W1_point(e1, e2)
    >>> e1, e2, e3 = e1.flatten(), e2.flatten(), e3.flatten()
    >>> green_surface = np.vstack((e1, e2, e3)).T
    """
    e1, e2 = np.atleast_1d(e1, e2)
    e3 = -2*(2*e1 - e2 - 1)*(4*e1**3 - 4*e1**2 - e1*e2 - e1 + e2**2 
                             + 2*e2 + (e1 - e2 - 1)*np.sqrt(-4*e1**2 + e2**2 
                                                            + 2*e2 + 1) + 1)\
        /(2*e1 - e2 + np.sqrt(-4*e1**2 + e2**2 + 2*e2 + 1) - 1)**2
    e3[(e1 == 0) & (e2 != 0)] = np.inf
    e3[(e1 == 0) & (e2 == 0)] = -1
    valid = valid_distribution(e1, e2, e3)
    return np.where(valid, e3, np.inf)


def W2_point(e1, e2):
    """
        Computes the values of `e3` for points on the W purple surface, given the
        correlators `e1` and `e2`.

        The function evaluates a mathematical expression to compute `e3` for points lying
        on the W purple surface. It also ensures that the computed `e3` values correspond
        to valid probability distributions by checking the `valid_distribution` function.
        If the distribution is invalid, the function returns `np.inf` for those points.

        Parameters:
        -----------
        e1 : float or numpy.ndarray
            The one-body correlator. Can be a single float or a numpy array of floats.
        e2 : float or numpy.ndarray
            The two-body correlator. Can be a single float or a numpy array of floats.

        Returns:
        --------
        numpy.ndarray
            An array of `e3` values corresponding to points on the W purple surface.
            If the computed `e3` values do not correspond to a valid probability
            distribution, the function returns `np.inf` for those points.

        Notes:
        ------
        - The `valid_distribution` function is used to check if the computed `e3` values
          correspond to valid probability distributions.
        - This function is not guaranteed to work for points on the surface boundary.

        Examples:
        ---------
        >>> # Example with floats
        >>> e1, e2 = 1/3, 1/9
        >>> e3 = W2_point(e1, e2)
        >>> print(f"Point on purple surface: {e1, e2, e3}")

        >>> # Example with numpy arrays
        >>> e1 = np.linspace(-1, 1, 30)
        >>> e2 = np.linspace(-1/3, 1, 20)
        >>> e1, e2 = np.meshgrid(e1, e2)
        >>> e3 = W2_point(e1, e2)
        >>> e1, e2, e3 = e1.flatten(), e2.flatten(), e3.flatten()
        >>> purple_surface = np.vstack((e1, e2, e3)).T
        """
    e1, e2 = np.atleast_1d(np.array(e1), np.array(e2))
    e3 = (68*e1**6 + 11*e1**5*e2 - 411*e1**5 - 4*e1**4*e2**2 - 123*e1**4*e2 + 795*e1**4 + 141*e1**3*e2**2 + 495*e1**3*e2 - 784*e1**3 - 557*e1**2*e2**2 - 407*e1**2*e2 + 364*e1**2 + 543*e1*e2**2 + 18*e1*e2 - 13*e1 - 99*e2**2 - 34*e2 + np.sqrt(2)*np.sqrt(1 - e1)*(-8*e1**6 - 2*e1**5*e2 + 140*e1**5 + 24*e1**4*e2 - 378*e1**4 - 25*e1**3*e2**2 - 203*e1**3*e2 + 430*e1**3 + 228*e1**2*e2**2 + 290*e1**2*e2 - 252*e1**2 - 349*e1*e2**2 - e1*e2 + 10*e1 + 70*e2**2 + 24*e2 + 2) - 3)/(-4*e1**5 - e1**4*e2 + 49*e1**4 + 33*e1**3*e2 - 109*e1**3 - 131*e1**2*e2 + 111*e1**2 + 131*e1*e2 - 51*e1 - 24*e2 + np.sqrt(2)*np.sqrt(1 - e1)*(-14*e1**4 - 6*e1**3*e2 + 52*e1**3 + 53*e1**2*e2 - 59*e1**2 - 84*e1*e2 + 38*e1 + 17*e2 + 3) - 4)
    valid = valid_distribution(e1, e2, e3)
    return np.where(valid, e3, np.inf)


def W3_point(e1, e2):
    """
    Computes the values of `e3` for points on the W red surface, given the
    correlators `e1` and `e2`.

    The function evaluates a mathematical expression to compute `e3` for points lying
    on the W red surface. It also ensures that the computed `e3` values correspond
    to valid probability distributions by checking the `valid_distribution` function.
    If the distribution is invalid, the function returns `np.inf` for those points.

    Parameters:
    -----------
    e1 : float or numpy.ndarray
        The one-body correlator. Can be a single float or a numpy array of floats.
    e2 : float or numpy.ndarray
        The two-body correlator. Can be a single float or a numpy array of floats.

    Returns:
    --------
    numpy.ndarray
        An array of `e3` values corresponding to points on the W red surface.
        If the computed `e3` values do not correspond to a valid probability
        distribution, the function returns `np.inf` for those points.

    Notes:
    ------
    - The `valid_distribution` function is used to check if the computed `e3` values
      correspond to valid probability distributions.
    - This function is not guaranteed to work for points on the surface boundary.

    Examples:
    ---------
    >>> # Example with floats
    >>> e1, e2 = 0.2, 0.2
    >>> e3 = W3_point(e1, e2)
    >>> print(f"Point on red surface: {e1, e2, e3}")

    >>> # Example with numpy arrays
    >>> e1 = np.linspace(-1, 1, 30)
    >>> e2 = np.linspace(-1/3, 1, 20)
    >>> e1, e2 = np.meshgrid(e1, e2)
    >>> e3 = W3_point(e1, e2)
    >>> e1, e2, e3 = e1.flatten(), e2.flatten(), e3.flatten()
    >>> red_surface = np.vstack((e1, e2, e3)).T
    """
    a = (1/2)*(2*e1 - e2 + ((-2*e1 + e2 + 1)*(e1**(3/2) + np.sqrt(e1) + np.sqrt(e1**3 + 2*e1**2 - e1 + e2 + 1)))**(2/3) - 1)/(np.sqrt(e1)*((-2*e1 + e2 + 1)*(e1**(3/2) + np.sqrt(e1) + np.sqrt(e1**3 + 2*e1**2 - e1 + e2 + 1)))**(1/3))
    b = -a*(4*a**2 + 4*a*e1 - 4*a - 2*e1 + e2 + 1)/(4*a**2 + 2*e1 - e2 - 1)
    e = (1 - 2*e1 + e2)/4/a**2
    f = (-1 + 4*a**2 + 2*e1 -e2)/8/a**2
    s = (1/4)*(-8*a**3*e1 + 20*a**2*e1 - 4*a**2*e2 - 8*a**2 - 12*a*e1 + 6*a*e2 + 6*a + 2*e1**2 - e1*e2 + e1 - e2 - 1)/(a**2*(2*e1 - e2 - 1))
    t = (1/4)*(4*a**2 + 2*e1 - e2 - 1)**2*(8*a**3*e1 - 4*a**2*e1 - 4*a**2*e2 - 4*a*e1 + 2*a*e2 + 2*a + 2*e1**2 - e1*e2 + e1 - e2 - 1)/(a**2*(2*e1 - e2 - 1)*(4*a**2 + 4*a*e1 - 4*a - 2*e1 + e2 + 1)**2)
    
    x = -16*e1**7 + 8*e1**6*e2 - 56*e1**6 + 32*e1**5*e2 - 144*e1**5 + 128*e1**4*e2 - 96*e1**4 - 20*e1**3*e2**2 + 152*e1**3*e2 + 84*e1**3 - 40*e1**2*e2**2 - 4*e1**2*e2 + 36*e1**2 - 14*e1*e2**2 - 28*e1*e2 - 14*e1 - e2**3 - 3*e2**2 - 3*e2 + 8*np.sqrt(e1*(e1 + 1)**2*(2*e1 - e2 - 1)**2*(e1**3 + 2*e1**2 - e1 + e2 + 1)**3) - 1
    e3 = (3/4)*(32*e1**4 - 16*e1**3*e2 + 48*e1**3 - 32*e1**2*e2 - 12*e1**2*x**(1/3) + 8*e1**2 + 4*e1*e2*x**(1/3) - 24*e1*e2 - 4*e1*x**(1/3) - 24*e1 + 2*e2**2 + 2*e2*x**(1/3) + 4*e2 + 2*x**(2/3) + 2*x**(1/3) + 2)/(x**(1/3)*(e1 + 1))
    return np.where((a>=0) & (b>=0) & (a+b<=1) & (e>=0) & (f>=0) & (e+f<=1) 
                    & (s>=0) & (s<=1) & (t>=0) & (t<=1), 
                    e3, np.inf)


def W4_point(e1, e2):
    """
    Computes the values of `e3` for points on the W yellow surface, given the
    correlators `e1` and `e2`.

    The function evaluates a mathematical expression to compute `e3` for points lying
    on the W yellow surface. It also ensures that the computed `e3` values correspond
    to valid probability distributions by checking the `valid_distribution` function.
    If the distribution is invalid, the function returns `np.inf` for those points.

    Parameters:
    -----------
    e1 : float or numpy.ndarray
        The one-body correlator. Can be a single float or a numpy array of floats.
    e2 : float or numpy.ndarray
        The two-body correlator. Can be a single float or a numpy array of floats.

    Returns:
    --------
    numpy.ndarray
        An array of `e3` values corresponding to points on the W yellow surface.
        If the computed `e3` values do not correspond to a valid probability
        distribution, the function returns `np.inf` for those points.

    Notes:
    ------
    - The `valid_distribution` function is used to check if the computed `e3` values
      correspond to valid probability distributions.
    - This function is not guaranteed to work for points on the surface boundary.

    Examples:
    ---------
    >>> # Example with floats
    >>> e1, e2 = 0.4, 0
    >>> e3 = W4_point(e1, e2)
    >>> print(f"Point on yellow surface: {e1, e2, e3}")

    >>> # Example with numpy arrays
    >>> e1 = np.linspace(-1, 1, 30)
    >>> e2 = np.linspace(-1/3, 1, 20)
    >>> e1, e2 = np.meshgrid(e1, e2)
    >>> e3 = W4_point(e1, e2)
    >>> e1, e2, e3 = e1.flatten(), e2.flatten(), e3.flatten()
    >>> yellow_surface = np.vstack((e1, e2, e3)).T
    """
    e1, e2 = np.broadcast_arrays(*np.atleast_1d(e1, e2))
    e3 = np.full_like(e1.flatten(), np.inf, float)
    for i, (e1_, e2_) in enumerate(zip(e1.flatten(), e2.flatten())):
        poly = Polynomial([4*e1_**3 - 4*e1_**2*e2_ + e1_*e2_**2 - 2*e1_*e2_ - 3*e1_ + e2_**2 + 2*e2_ + 1, 
                           e1_**5 - 5*e1_**4 + 3*e1_**3*e2_ - 13*e1_**3 + 13*e1_**2*e2_ - 3*e1_**2 - 3*e1_*e2_**2 + 11*e1_*e2_ + 17*e1_ - 5*e2_**2 - 11*e2_ - 5, 
                           8*e1_**4 - 4*e1_**3*e2_ + 20*e1_**3 - 16*e1_**2*e2_ + 16*e1_**2 + 3*e1_*e2_**2 - 26*e1_*e2_ - 41*e1_ + 9*e2_**2 + 22*e2_ + 9, 
                           -4*e1_**4 - 16*e1_**3 + 8*e1_**2*e2_ - 32*e1_**2 - e1_*e2_**2 + 30*e1_*e2_ + 51*e1_ - 7*e2_**2 - 22*e2_ - 7, 
                           32*e1_**2 - 16*e1_*e2_ - 32*e1_ + 2*e2_**2 + 12*e2_ + 2, 
                           4*e1_**3 - 12*e1_**2 + 4*e1_*e2_ + 8*e1_ - 4*e2_])
        c = poly.roots()
        c = c[~np.iscomplex(c)].real
        a = (2*e1_ - e2_ - 1)*(2*c**2 - 2*c*e1_ - 2*c + e1_ + 1)/(8*c**2*e1_ - 4*c*e1_**2 - 8*c*e1_ + 2*c*e2_ + 2*c + 4*e1_ - 2*e2_ - 2)
        b = -a*(4*a**2*c*e1_ + 4*a**2*c - 4*a*c**2*e1_ + 4*a*c**2 - 8*a*c - 4*a*e1_ + 2*a*e2_ + 2*a + 4*c**2*e1_ - 4*c**2 - 4*c*e1_**2 + 4*c + 2*e1_**2 - e1_*e2_ + e1_ - e2_ - 1)/(4*a**2*c*e1_ + 4*a**2*c - 4*a*c**2*e1_ + 4*a*c**2 - 4*a*c*e1_ - 4*a*c + 4*c*e1_ - 2*c*e2_ - 2*c - 2*e1_**2 + e1_*e2_ - e1_ + e2_ + 1)
        d = -c*(4*a**2*c*e1_ - 4*a**2*c - 4*a**2*e1_ + 4*a**2 - 4*a*c**2*e1_ - 4*a*c**2 + 8*a*c + 4*a*e1_**2 - 4*a + 4*c*e1_ - 2*c*e2_ - 2*c - 2*e1_**2 + e1_*e2_ - e1_ + e2_ + 1)/(4*a**2*c*e1_ - 4*a**2*c - 4*a*c**2*e1_ - 4*a*c**2 + 4*a*c*e1_ + 4*a*c - 4*a*e1_ + 2*a*e2_ + 2*a + 2*e1_**2 - e1_*e2_ + e1_ - e2_ - 1)
        e = (1/4)*(-2*e1_ + e2_ + 1)/(a*c)
        f = (1/8)*(4*a**2*c*e1_ + 4*a**2*c - 4*a*c**2*e1_ + 4*a*c**2 - 4*a*c*e1_ - 4*a*c + 4*c*e1_ - 2*c*e2_ - 2*c - 2*e1_**2 + e1_*e2_ - e1_ + e2_ + 1)/(a*c*(a + c - e1_ - 1))
        s = (1/4)*(16*a**2*c**2*e1_ - 4*a**2*c*e1_**2 - 24*a**2*c*e1_ + 4*a**2*c*e2_ + 8*a**2*c + 8*a**2*e1_ - 4*a**2*e2_ - 4*a**2 - 4*a*c**2*e1_**2 - 24*a*c**2*e1_ + 4*a*c**2*e2_ + 8*a*c**2 + 20*a*c*e1_**2 - 4*a*c*e1_*e2_ + 20*a*c*e1_ - 8*a*c*e2_ - 12*a*c - 8*a*e1_**2 + 4*a*e1_*e2_ - 4*a*e1_ + 4*a*e2_ + 4*a + 8*c**2*e1_ - 4*c**2*e2_ - 4*c**2 - 8*c*e1_**2 + 4*c*e1_*e2_ - 4*c*e1_ + 4*c*e2_ + 4*c + 2*e1_**3 - e1_**2*e2_ + 3*e1_**2 - 2*e1_*e2_ - e2_ - 1)/(a*c*(2*e1_ - e2_ - 1)*(-a - c + e1_ + 1))
        valid = ((a >= 0) & (b >= 0) & (a + b <= 1) & (c >= 0) & (d>= 0) 
                 & (c + d <= 1) & (e >= 0) & (f.real >= 0) & (e + f <= 1) 
                 & (s >= 0) & (s <= 1))
        if valid.sum() != 0:
            e3[i] = np.min((1/2)*(16*a**2*c**2*e1_ - 4*a**2*c*e1_**2 - 22*a**2*c*e1_ + 6*a**2*c*e2_ + 6*a**2*c + 8*a**2*e1_ - 4*a**2*e2_ - 4*a**2 - 4*a*c**2*e1_**2 - 22*a*c**2*e1_ + 6*a*c**2*e2_ + 6*a*c**2 + 18*a*c*e1_**2 - 6*a*c*e1_*e2_ + 20*a*c*e1_ - 10*a*c*e2_ - 10*a*c - 8*a*e1_**2 + 4*a*e1_*e2_ - 4*a*e1_ + 4*a*e2_ + 4*a + 8*c**2*e1_ - 4*c**2*e2_ - 4*c**2 - 8*c*e1_**2 + 4*c*e1_*e2_ - 4*c*e1_ + 4*c*e2_ + 4*c + 2*e1_**3 - e1_**2*e2_ + 3*e1_**2 - 2*e1_*e2_ - e2_ - 1)/(a*c*(a + c - e1_ - 1)))
    return e3.reshape(e1.shape)


def W5_point(e1, e2):
    """
    Computes the values of `e3` for points on the W cyan surface, given the
    correlators `e1` and `e2`.

    The function evaluates a mathematical expression to compute `e3` for points lying
    on the W cyan surface. It also ensures that the computed `e3` values correspond
    to valid probability distributions by checking the `valid_distribution` function.
    If the distribution is invalid, the function returns `np.inf` for those points.

    Parameters:
    -----------
    e1 : float or numpy.ndarray
        The one-body correlator. Can be a single float or a numpy array of floats.
    e2 : float or numpy.ndarray
        The two-body correlator. Can be a single float or a numpy array of floats.

    Returns:
    --------
    numpy.ndarray
        An array of `e3` values corresponding to points on the W cyan surface.
        If the computed `e3` values do not correspond to a valid probability
        distribution, the function returns `np.inf` for those points.

    Notes:
    ------
    - The `valid_distribution` function is used to check if the computed `e3` values
      correspond to valid probability distributions.
    - This function is not guaranteed to work for points on the surface boundary.

    Examples:
    ---------
    >>> # Example with floats
    >>> e1, e2 = 1/3, -5/27
    >>> e3 = W5_point(e1, e2)
    >>> print(f"Point on cyan surface: {e1, e2, e3}")

    >>> # Example with numpy arrays
    >>> e1 = np.linspace(-1, 1, 30)
    >>> e2 = np.linspace(-1/3, 1, 20)
    >>> e1, e2 = np.meshgrid(e1, e2)
    >>> e3 = W5_point(e1, e2)
    >>> e1, e2, e3 = e1.flatten(), e2.flatten(), e3.flatten()
    >>> cyan_surface = np.vstack((e1, e2, e3)).T
    """
    e1, e2 = np.broadcast_arrays(*np.atleast_1d(e1, e2))
    e3 = np.full_like(e1.flatten(), np.inf, float)
    
    # intersection with D0 W Wflip plane
    e = np.linspace(1/2, 1/3, 40)
    border1 = np.vstack((e, (2*np.sqrt(2*(3*e-1))*(1-3*e) + 36*e - 17)/27)).T
    params1 = np.vstack(((2 + np.sqrt(2*(3*e-1)))/6, (1 - np.sqrt(2*(3*e-1)))/3)).T
    
    # intersection with red surface
    e = np.linspace(1/3, 1, 40)
    a = np.empty_like(e)
    for i, e_ in enumerate(e):
        a[i] = Polynomial([4*e_**2*(1 + e_)**2, 
                           -5*e_ - 16*e_**2 - 50*e_**3 - 24*e_**4 - e_**5, 
                           1 + 12*e_ + 34*e_**2 + 84*e_**3 + 45*e_**4, 
                           -12*e_ - 24*e_**2 - 60*e_**3, 
                           16*e_**2]).roots()[0].real
    border2 = np.vstack(((1-3*a)*e/(a-e), 
                         (-a + 3*e - 6*a*e + 4*a**3*e - 4*a**2*e**2)/(a - e))).T
    params2 = np.vstack((a, a)).T
    border = np.vstack((border1, border2))
    params = np.vstack((params1, params2))
    
    for i, (e1_, e2_) in enumerate(zip(e1.flatten(), e2.flatten())):
        
        if (e1_ > 1/2 or e1_ < np.sqrt(5)-2 
                or e2_ > 9-4*np.sqrt(5) or e2_ < -5/27 
                or e2_ < (-5 + 2*(3*e1_-1)*(6 - np.sqrt(2*(3*e1_-1))))/27):
            continue
        
        distances_square = np.sum((border - np.array([e1_, e2_]))**2, axis=1)
        initial_point = border[np.argmin(distances_square)]
        initial_params = params[np.argmin(distances_square)]
        displacement = np.array([e1_, e2_]) - initial_point
        distance = np.sqrt(np.sum(displacement**2))
        direction = displacement/distance
        step = 0.003
        alphas = np.arange(0, distance, step)
        x0 = initial_params
        
        for alpha in alphas:
            point = initial_point + alpha*direction
            solution = root(fun, x0, args=point, jac=jac)
            x0 = solution.x
        
        solution = root(fun, x0, args=((e1_, e2_), ), jac=jac)
        a, c = solution.x
        b = bfun(a, c, e1_, e2_)
        d = dfun(a, c, e1_, e2_)
        e = efun(a, c, e1_, e2_)
        f = ffun(a, c, e1_, e2_)
        s = sfun(a, c, e1_, e2_)
        t = tfun(a, c, e1_, e2_)
        e3_ = e3fun(a, c, e1_, e2_)
        if (a>=0 and b>=0 and a+b<=1 and c>=0 and d>=0 and c+d<=1 and e>=0 
                and f>=0 and e+f<=1 and s>=0 and s<=1 
                and np.isclose(t, 1, rtol=1e-3) and e3_ < e3[i]):
            e3[i] = e3fun(a, c, e1_, e2_)
    return e3


def W5_point_fixed_initial_guesses(e1, e2):
    e1, e2 = np.broadcast_arrays(*np.atleast_1d(e1, e2))
    e3 = np.full_like(e1.flatten(), np.inf)
    initial_guesses = np.array([[0.28, 0.44], 
                                [0.36, 0.46], 
                                [0.32, 0.3], 
                                [0.3, 0.5], 
                                [0.4, 0.44]])
    
    for i, (e1_, e2_) in enumerate(zip(e1.flatten(), e2.flatten())):
        
        if e1_ > 1/2 or e1_ < np.sqrt(5)-2:
            continue
        
        for x0 in initial_guesses:
            solution = root(fun, x0, args=((e1_, e2_), ), method='hybr', jac=jac)
            a, c = solution.x
            b = bfun(a, c, e1_, e2_)
            d = dfun(a, c, e1_, e2_)
            e = efun(a, c, e1_, e2_)
            f = ffun(a, c, e1_, e2_)
            s = sfun(a, c, e1_, e2_)
            t = tfun(a, c, e1_, e2_)
            e3_ = e3fun(a, c, e1_, e2_)
            if (a>=0 and b>=0 and a+b<=1 and c>=0 and d>=0 and c+d<=1 and e>=0 
                    and f>=0 and e+f<=1 and s>=0 and s<=1 
                    and np.isclose(t, 1, rtol=1e-3) and e3_ < e3[i]):
                e3[i] = e3fun(a, c, e1_, e2_)
    return e3


# ---------------------------------------- Generate points on boundary (random)
def choose_W1_point():
    a = np.linspace(0,0.19158750185069973, 1000)
    p = (-1 + 2*a + np.sqrt(5 - 8*a + 4*a**2))/2 - (1+a)/4 \
        - 1/4*np.sqrt((1+5*a**2-2*a**3)/(1-2*a))
    p = p/p.sum()
    a = np.random.choice(a, p=p)
    b = np.random.uniform((1+a)/4 + 1/4*np.sqrt((1+5*a**2-2*a**3)/(1-2*a)), 
                          (-1 + 2*a + np.sqrt(5 - 8*a + 4*a**2))/2)
    e1 = -4*a*b + 2*a + 2*b - 1
    e2 = -8*a*b**2 + 8*a*b - 4*a + 4*b**2 - 4*b + 1
    e3 = 16*a**2*b - 8*a**2 - 8*a*b**2 - 4*a*b + 6*a + 4*b**2 - 2*b - 1
    return e1, e2, e3


def choose_W3_point():
    e = np.linspace(1/2, 0.642129856069591, 1000)
    dmin = np.fmax((2*e**2 + np.sqrt(8*e**4 + 4*e**3 - 12*e**2 - 4*e 
                                     + 5) - 1)/(1-e)/2, 
                   (e**2 + e - 1)/(2*e - 1))
    dmax = (-e**3 - e + 1)/(e**2 - 3*e + 2)
    p = dmax - dmin
    p = p/p.sum()
    e = np.random.choice(e, p=p)
    dmin = np.fmax((2*e**2 + np.sqrt(8*e**4 + 4*e**3 - 12*e**2 - 4*e 
                                     + 5) - 1)/(1-e)/2, 
                   (e**2 + e - 1)/(2*e - 1))
    dmax = (-e**3 - e + 1)/(e**2 - 3*e + 2)
    d = np.random.uniform(dmin, dmax)
    e1 = 1 - 2*e**2
    e2 = -8*d*e**2 + 16*d*e - 8*d - 8*e + 5
    e3 = -16*d*e**2 + 24*d*e - 8*d + 8*e**3 + 6*e**2 - 16*e + 5
    return e1, e2, e3


def choose_W4_point():
    d = np.linspace(0, 1/2, 1000) + 0j
    emin = (2/3)*d - 1/3*2**(1/3)*(-7*d**2 + 8*d - 4)/(34*d**3 - 93*d**2 + 21*d + 3*np.sqrt(3)*np.sqrt(-8*d**6 - 60*d**5 + 87*d**4 + 158*d**3 - 223*d**2 + 74*d - 5) + 11)**(1/3) + (1/6)*2**(2/3)*(34*d**3 - 93*d**2 + 21*d + 3*np.sqrt(3)*np.sqrt(-8*d**6 - 60*d**5 + 87*d**4 + 158*d**3 - 223*d**2 + 74*d - 5) + 11)**(1/3) - 2/3
    d, emin = d.real, emin.real
    emax = 1-d
    p = emax - emin
    p = p/p.sum()
    d = np.random.choice(d, p=p) + 0j
    emin = (2/3)*d - 1/3*2**(1/3)*(-7*d**2 + 8*d - 4)/(34*d**3 - 93*d**2 + 21*d + 3*np.sqrt(3)*np.sqrt(-8*d**6 - 60*d**5 + 87*d**4 + 158*d**3 - 223*d**2 + 74*d - 5) + 11)**(1/3) + (1/6)*2**(2/3)*(34*d**3 - 93*d**2 + 21*d + 3*np.sqrt(3)*np.sqrt(-8*d**6 - 60*d**5 + 87*d**4 + 158*d**3 - 223*d**2 + 74*d - 5) + 11)**(1/3) - 2/3
    d, emin = d.real, emin.real
    emax = 1-d
    e = np.random.uniform(emin, emax)
    c = np.roots([e**2, 
                  -e+2*d*e-e**2+d*e**2, 
                  -d+d**2+e-3*d*e+3*d**2*e+d*e**2-d**2*e**2-d*e**3, 
                  d-3*d**2+2*d**3-d*e+3*d**2*e-d**3*e+d*e**2-2*d**2*e**2])
    c = np.sort(c)[1]
    e1 = -1 + 2*d + 2*c*e
    e2 = (2*c**3*e**2 + 6*c**2*d*e**2 - 2*c**2*e**2 - c**2*e + 6*c*d**2*e**2 - 2*c*d**2*e + 2*c*d**2 - 2*c*d*e**3 - 2*c*d*e**2 - c*d*e - c*d + c*e + 4*d**3*e**2 - 6*d**3*e + 4*d**3 - 4*d**2*e**2 + 5*d**2*e - 4*d**2 + d*e**2 - d*e + d)/(c**2*e + c*d*e + c*d - c*e - d**2*e + 2*d**2 - d*e**2 + d*e - d)
    e3 = (2*c**3*e**2 + 6*c**2*d*e**2 - 2*c**2*e**2 - c**2*e + 14*c*d**2*e**2 - 10*c*d**2*e + 2*c*d**2 - 2*c*d*e**3 - 2*c*d*e**2 - c*d*e - c*d + c*e + 12*d**3*e**2 - 14*d**3*e + 4*d**3 - 10*d**2*e**2 + 11*d**2*e - 4*d**2 + d*e**2 - d*e + d)/(c**2*e + c*d*e + c*d - c*e - d**2*e + 2*d**2 - d*e**2 + d*e - d)  
    if W5(e1, e2, e3) >= 0:
        e1, e2, e3 = choose_W4_point()
    return e1, e2, e3


def choose_W5_point():
    d = np.linspace(1/2, 1, 1000)
    qmin = (d<0.642129856069591)*(3*d**4 - 3*d**3 + 2*d - 1)/(d**4 + d**2 - d)
    p = 1 - qmin
    p = p/p.sum()
    d = np.random.choice(d, p=p)
    qmin = (d<0.642129856069591)*(3*d**4 - 3*d**3 + 2*d - 1)/(d**4 + d**2 - d)
    q = np.random.uniform(qmin, 1)
    e1 = 1 - 2*d**2
    e2 = (-8*d**6 + 8*d**5*q + 16*d**5 - 12*d**4*q - 12*d**4 + 5*d**3*q + d**2*q + 6*d**2 - d*q - 4*d + 1)/(d**3*q - 4*d**3 + d**2*q + 6*d**2 - d*q - 4*d + 1)
    e3 = (-8*d**6*q + 30*d**5*q - 38*d**4*q + 15*d**3*q + d**2*q + 4*d**2 - d*q - 4*d + 1)/(d**3*q - 4*d**3 + d**2*q + 6*d**2 - d*q - 4*d + 1)
    if W1(e1, e2, e3) >= 0 or W3(e1, e2, e3) >= 0 or W4(e1, e2, e3) >= 0:
        e1, e2, e3 = choose_W5_point()
    return e1, e2, e3


def choose_GHZ_point():
    a = np.random.triangular(0, 0, 1)
    b = np.random.uniform(0, 1-a)
    e1 = 4*a*b + 2*b**2 -1
    e2 = 4*a**2*b + 12*a*b**2 - 8*a*b + 4*b**3 - 4*b**2 + 1
    e3 = -12*a**2*b - 12*a*b**2 + 12*a*b - 4*b**3 + 6*b**2 - 1
    if GHZ(e1, e2, e3, flip=True) >= 0:
        e1, e2, e3 = choose_GHZ_point()
    return e1, e2, e3
