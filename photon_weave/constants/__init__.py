import numpy as np

# CONSTANTS
C0 = 299792458  # m/s


# FUNCTIONS
def gaussian(
    t: float, t_a: float, omega: float, mu: float = 0, sigma: float = 1
) -> float:
    # norm = 1/(sigma*np.sqrt(2 * np.pi))
    norm = 1 / (np.sqrt(sigma * np.pi))
    exponent = -((t - t_a - mu) ** 2) / (2 * sigma**2)
    envelope = norm * np.exp(exponent)
    # carrier = np.exp(1j * omega * t)
    return envelope
