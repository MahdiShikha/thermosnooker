"""Physics Module
Contains the maxwell pde function which returns the probabiltiy
of a ball having a certain speed.
"""
import numpy as np
def maxwell(speed,kbt,mass=1.):
    """Computes the probability of a ball having a certain speed
    based on the Maxwell-Boltzmann Distribution

    Args:
        speed (List or NDArray[float, 2x1]): Speeds to evaluate at
        kbt (float): Thermodynamic temperature of balls
        mass: Mass of balls

    Returns:
        NDArray[float]: The probability density of each speed.
        If kbt == 0, returns an array of zeros.

    """
    # if kbt == 0:
    #     raise RuntimeError("Cannot have kbt = 0")
    if kbt == 0:
        return np.zeros_like(speed)
    constant  = mass/kbt
    exponent = (-1/2*mass*speed**2)/(kbt)
    return constant*speed*np.exp(exponent)
