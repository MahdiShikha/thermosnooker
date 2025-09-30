"""Ideal Gas Law Analysis Module
"""
import matplotlib.pyplot as plt
import numpy as np
from thermosnooker.balls import Ball,Container
from thermosnooker.simulations import MultiBallSimulation
from thermosnooker.physics import maxwell

K_B = 1.380649e-23

def IGL_comparision():

    """Ideal Gas Law Comparision.

    In this function we investigate how well our simulation reproduces the distributions of the IGL.
    Creates 3 figures, namely:
    1) Pressure vs Temperature plot
    2) Pressure vs Volume plot
    3) Pressure vs Number of Balls plot
    Ensure that this function returns the three matplotlib figures.

    Returns:
        tuple[Figure, Figure, Figure]: The 3 requested figures: (PT, PV, PN)
    """
    #Pressure vs Temperature
    fig_pt,ax_pt = plt.subplots()
    ax_pt.set(xlabel=r"Temperature $K_{B}$T",
              ylabel="Pressure (Pa)")
    ball_radii = [0.5,1.0]
    speeds = np.logspace(-1,np.log10(300),num=20,endpoint=True,base=10,dtype=float)

    for radii in ball_radii:
        pressures = []
        temps = []
        for v in speeds:
            mbs = MultiBallSimulation(b_radius=radii,b_speed=v)
            t_eq = mbs.t_equipartition()
            mbs.run(800,animate=False)
            pressures.append(mbs.pressure())
            temps.append(t_eq)

        temps = np.array(temps,dtype=float)
        pt = np.array(pressures,dtype=float)

        ax_pt.scatter(temps,pt,marker="x",label=f"Simulated at (r = {radii}m)")
        y_igl = temps*len(mbs.balls())*K_B/(mbs.container().volume())

    ax_pt.plot(temps,y_igl,color="Green",linestyle="dashed",
               label="Ideal Gas prediction")
    ax_pt.legend(loc="best")
    fig_pt.tight_layout()

    #Pressure vs Volume
    fig_pv,ax_pv = plt.subplots()
    ax_pv.set(xlabel=r"Volume $(m^{3})$",ylabel="Pressure (Pa)")
    container_radii = np.linspace(10,20,num=21,endpoint=True)

    v_0 = 10.0
    for radii in ball_radii:
        pressures = []

        mbs0 = MultiBallSimulation(b_radius=radii,b_speed=v_0)
        t0 = mbs0.t_equipartition()
        for r in container_radii:
            mbs = MultiBallSimulation(b_radius=radii,c_radius = r)
            mbs.run(800,animate=False)
            pressures.append(mbs.pressure())

        y_igl = t0*len(mbs.balls())*K_B/(container_radii**2*np.pi)
        ax_pv.scatter(container_radii**2*np.pi,pressures,marker="x",
                      label=f"Simulated at (r = {radii}m)")

    ax_pv.plot(container_radii**2*np.pi,y_igl,color="Green",linestyle="dashed",
               label="Ideal Gas Prediction)")

    ax_pv.legend(loc="best")
    fig_pv.tight_layout()

    #Pressure vs Ball number
    fig_pn,ax_pn = plt.subplots()
    ax_pn.set(xlabel="Number of Balls",ylabel="Pressure (Pa)")
    N_vals = np.linspace(1,37,num=37,endpoint=True,dtype=int)

    V = MultiBallSimulation().container().volume()
    for radii in ball_radii:
        pressures = []
        for N in N_vals:
            mbs = MultiBallSimulation(b_radius=radii)
            mbs._balls = mbs.balls()[:N]
            if len(mbs._balls) != N:
                raise RuntimeError(f"Simulation did not generate expected ball count {N}, instead generated {len(mbs.balls())}")
            mbs.run(800,animate=False,pause_time=0.01)
            pressures.append(mbs.pressure())

        y_igl = N_vals*t0*K_B/V
        ax_pn.scatter(N_vals,pressures,marker="x",label=f"Simulated at (r = {radii}m)")

    ax_pn.plot(N_vals,y_igl,color="Green",linestyle="dashed",label="Ideal Gas prediction")
    ax_pn.legend(loc="best")
    fig_pn.tight_layout()

    return fig_pt,fig_pv,fig_pn

def Boltzman_comparision():
    """Boltzmann Distribution Comparision

    In this function we shall plot a histogram to investigate how the speeds of
    the balls evolve from the initial value. We shall then compare this to the
    Maxwell-Boltzmann distribution. Ensure that this function returns the
    created histogram.

    Returns:
        Figure: The speed histogram.
    """
    inital_speeds = [10,20,30]
    fig,ax = plt.subplots()
    colours = ["blue","green","red"]
    j = 0
    for vel in inital_speeds:
        mbs = MultiBallSimulation(b_speed=vel)
        mbs.run(2000,animate=False)
        t_ideal = mbs.t_ideal()

        v_max = np.array(mbs.speeds()).max()
        v_values = np.linspace(0.1,v_max*1.75,num=2000)
        counts,bins,patches = ax.hist(mbs.speeds(),density=True,color = colours[j],alpha=0.4,
                                      bins=10,label="Simulated values of speed")
        mass = mbs.balls()[0].mass()

        pdf_values = maxwell(v_values,K_B*t_ideal,mass)
        ax.plot(v_values,pdf_values,linestyle="dashed",
                color = colours[j],
                label=f"2D Maxwell Distribution at (K_BT = {t_ideal*K_B:.1f}K_B, v_0 = {vel}ms^-1")
        j = j+1
    ax.set_xlabel(r"Speeds $(ms^-1)$")
    ax.set_ylabel("Probability of speed P(v)")
    ax.set_xlim(right=bins[-1]+5)
    ax.legend(fontsize="small")



    return fig

if __name__ == "__main__":
    FIG_PT, FIG_PV, FIG_PN = IGL_comparision()
    Boltzmann_Fig = Boltzman_comparision()

plt.show()