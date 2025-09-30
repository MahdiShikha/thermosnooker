"""Brownian Motion Demo Module
"""
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.collections import LineCollection
from thermosnooker.simulations import BrownianSimulation

def demo_brownian_motion():

    """Brownian Motion Demo.

    In this function we shall run a Brownian motion simulation
    and plot the resulting trajectory of the 'big' ball.

    Returns:
        [figure]: A figure showing the trajectory of the big ball
                  super imposed on the finale state of the system.
    """
    brown_mbs = BrownianSimulation()
    brown_mbs.run(5000,animate=False)
    fig,ax = brown_mbs.setup_figure()

    positions = np.array(brown_mbs.bb_positions())
    times = np.array(brown_mbs.bb_times())

    points = positions.reshape(-1,1,2)
    line_segments = np.concatenate([points[:-1],points[1:]],axis=1)

    norm = plt.Normalize(times[0],times[-1])
    lc = LineCollection(line_segments,cmap="cividis",norm=norm,linewidth=4)
    lc.set_array(times[:-1])
    ax.add_collection(lc)
    fig.colorbar(lc,ax=ax,label="time (s)")

    return fig
if __name__ == "__main__":
    demo_brownian_motion()

plt.show()