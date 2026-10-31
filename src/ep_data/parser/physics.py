import numpy as np

MASS_PROTON = 0.9382720813
MASS_KAON = 0.493677
MASS_LAMBDA = 1.115683
MASS_SIGMA0 = 1.192642


def get_dOmega_dt(
    Q2: np.ndarray,
    W: np.ndarray,
    hyperon_mass: float = MASS_LAMBDA,
):
    """
    Calculate the differential solid angle dOmega/dt for the given Q2 and W.

    Parameters
    ----------
    Q2 : np.ndarray
        The squared four-momentum transfer.
    W : np.ndarray
        The invariant mass of the final state.
    hyperon_mass : float
        The final-state hyperon mass in GeV.

    Returns
    -------
    dOmega_dt : np.ndarray
        The differential solid angle dOmega/dt.
    """

    E_gamma = (W**2 - MASS_PROTON**2 - Q2) / (2 * W)
    qmag = np.sqrt(E_gamma**2 + Q2)
    E_K = (W**2 + MASS_KAON**2 - hyperon_mass**2) / (2 * W)
    pKmag = np.sqrt(E_K**2 - MASS_KAON**2)
    dOmega_dt = np.pi / (pKmag * qmag)
    return dOmega_dt


def get_dt_dOmega(
    Q2: np.ndarray,
    W: np.ndarray,
    hyperon_mass: float = MASS_LAMBDA,
):
    """Calculate the inverse differential solid angle for a hyperon mass."""
    return 1.0 / get_dOmega_dt(Q2, W, hyperon_mass)


def get_x_from_Q2_W(
    Q2: np.ndarray,
    W: np.ndarray,
    Q2_err: np.ndarray | None = None,
    W_err: np.ndarray | None = None,
) -> np.ndarray:
    """
    Calculate the Bjorken x from Q2 and W.
    """
    x = Q2 / (Q2 + W**2 - MASS_PROTON**2)
    if Q2_err is not None and W_err is not None:
        x_err = np.abs(x) * np.sqrt((Q2_err / Q2) ** 2 + (W_err / W) ** 2)
        return x, x_err
    return x


def get_W_from_Q2_x(
    Q2: np.ndarray,
    x: np.ndarray,
    Q2_err: np.ndarray | None = None,
    x_err: np.ndarray | None = None,
) -> np.ndarray:
    """
    Calculate the invariant mass W from Q2 and Bjorken x.
    """
    W = np.sqrt(MASS_PROTON**2 + Q2 * (1 - x) / x)
    if Q2_err is not None and x_err is not None:
        W_err = np.abs(W) * np.sqrt((Q2_err / Q2) ** 2 + (x_err / (1 - x)) ** 2)
        return W, W_err
    return W
