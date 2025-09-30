"""MultiBall Sim Demo Module
"""
import matplotlib.pyplot as plt
import numpy as np
from thermosnooker.balls import Ball,Container
from thermosnooker.simulations import MultiBallSimulation


def demo_multi_ball():
    """Multi Ball Simulation.

    Creates an instance and animates 1000 collisions with default parameters
    """
    mbs = MultiBallSimulation()
    mbs.run(1000,animate=True,pause_time=0.005)



if __name__ == "__main__":
    demo_multi_ball()

plt.show()