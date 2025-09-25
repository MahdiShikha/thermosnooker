"""Simulations Module
Contains the base and specific Simulation classes
for 2D elastic-collision systems
"""
# pylint: disable=import-error
import matplotlib.pyplot as plt
import numpy as np
from thermosnooker.balls import Ball, Container
K_B = 1.380649e-23

def rtrings(rmax,nrings,multi):
    """Generate polar co-ordinates on rings up to a max radius
    Yields the origin and then for each ring, nr=i*multi angles
    between 0 and 2π.

    Args:
        rmax (float): radius of the outermost ring
        nrings (int): number of rings
        multi (int): points per ring multiplier

    Yields:
        tuple[float, float]: (radius, angle) coordinates for each point
    """
    yield (0.0,0.0)
    r = np.linspace(0,rmax,nrings+1)
    for i in range(1,nrings+1):
        radius = r[i]
        theta = np.linspace(0,2*np.pi,i*multi,endpoint=False) #important to include endpoint=False
        for j in theta:
            yield (radius,j)


class Simulation:
    """Base class for collision simulations."""

    def next_collision(self):
        """Check if next collision method is implemented."""
        raise NotImplementedError('next_collision() needs to be implemented in derived classes')
    def setup_figure(self):
        """Check if next figure method is implemented."""
        raise NotImplementedError('setup_figure() needs to be implemented in derived classes')
    def run(self, num_collisions, animate=False, pause_time=0.001):
        """Run the simulation for a number of collisions
        Args:
            num_collisions (int): how many collisions to be simulated
            animate (bool): if the simulation should be animated
            pause_time (float): how long each frame should last for
        """
        if animate:
            fig, axes = self.setup_figure()
        for _ in range(num_collisions):
            self.next_collision()
            if animate:
                plt.pause(pause_time)
        if animate:
            plt.show()

class SingleBallSimulation(Simulation):
    """Simulation of a single ball colliding elasitcally within a container"""

    def __init__(self,container,ball):
        """Initalise a single ball simulation

        Args:
            container (Container): Container enclosing ball
            ball (Ball): single moving ball
        """
        self._container = container
        self._ball = ball

    def container(self):
        """Get the container object

        Returns:
            container: Container object
        """
        return self._container

    def ball(self):
        """Get the ball object

        Returns:
            ball: Ball object"""
        return self._ball

    def setup_figure(self):
        """Draw the container and ball on a new figure

        Returns:
            tuple[Figure, Axes]: The figure and axis for animation in sim
        """
        rad = self._container.radius()
        fig = plt.figure()
        ax = plt.axes(xlim=(-rad, rad), ylim=(-rad, rad))
        ax.set_aspect("equal", "box")
        ax.add_artist(self.container().patch())
        ax.add_patch(self.ball().patch())
        return fig, ax

    def next_collision(self):
        """Progresses the simulation to the next collision

        Returns:
            float: time until the next collision"""
        t = self._container.time_to_collision(self._ball)
        self._container.move(t)
        self._ball.move(t)
        self._container.collide(self._ball)
        return t

class MultiBallSimulation(Simulation):
    """Simulation of many balls of equal radius in a container"""
    def __init__(self,
    c_radius=10,b_radius=1,b_speed=10.,b_mass=1,rmax=8,nrings=3,multi=6):
        """Initalise balls in a container based on rtrings generation

        Args:
            c_radius (float): container radius
            b_radius (float): radius of balls
            b_speed (float): inital ball speed
            b_mass (float): ball mass
            rmax (float): radius of outermost ring for intialisation
            nrings (int): number of rings
            multi (int): points per ring multiplier
        """
        self._container_radius = c_radius
        self._container = Container(self._container_radius)
        self._balls = []
        self._time = 0.
        self._kinetic_energy = []
        self._momentum = []
        self._pressure = []
        for r,theta in rtrings(rmax,nrings,multi):
            x = r*np.cos(theta)
            y = r*np.sin(theta)

            rng = np.random.default_rng()
            angle = rng.random()*2*np.pi
            vel_x = b_speed * np.cos(angle)
            vel_y = b_speed * np.sin(angle)

            b = Ball([x,y],[vel_x,vel_y],b_radius,b_mass)
            self._balls.append(b)


    def container(self):
        """Get the container object

        Returns:
            container: Container object
        """
        return self._container

    def balls(self):
        """Get the ball object

        Returns:
            ball: Ball object"""
        return self._balls

    def setup_figure(self):
        """Draw container and balls on a new figure

        Returns:
            tuple[Figure, Axes]: The figure and axisfor animation in sim
        """
        rad = self._container.radius()
        fig = plt.figure()
        ax = plt.axes(xlim=(-rad, rad), ylim=(-rad, rad))
        ax.set_aspect("equal", "box")
        ax.add_artist(self.container().patch())
        for ball in self._balls:
            ax.add_patch(ball.patch())
        return fig, ax

    def next_collision(self):
        """Find the next time to collision for all objects in the simulation

        Computes the time to collision method for all Ball objects in the
        Multi Sim class, including the container.

        Returns:
            float: time to the next collision
        """
        event_info = []
        #wall on ball collisions
        for i in (self._balls):
            t = self._container.time_to_collision(i)
            if isinstance(t,float) is True:
                event_info.append((t,self._container,i))

        #ball on ball collisions
        for j, ball_1 in enumerate(self._balls):
            for k in range(j+1,len(self._balls)):
                ball_2 = self._balls[k]
                t = ball_1.time_to_collision(ball_2)
                if isinstance(t,float):
                    event_info.append((t,ball_1,ball_2))

        #picking the next collision
        t_min,object_1,object_2 = min(event_info)
        self._container.move(t_min)
        for l in self._balls:
            l.move(t_min)
        object_1.collide(object_2)
        self._time += t_min
        self._kinetic_energy.append((self._time,self.kinetic_energy(include_container=True)))
        self._momentum.append((self._time,self.momentum(include_container=True)))
        self._pressure.append((self._time,self.pressure()))
        return t_min

    def kinetic_energy(self,include_container=True):
        """Compute the total kinetic energy of all balls (and container).

        Args:
            include_container (bool): Option to include the KE of the container

        Returns:
            float: total KE
        """
        tot_ke = 0.
        for b in self._balls:
            ke = 0.5*b.mass()*(np.dot(b.vel(),b.vel()))
            tot_ke += ke
        if include_container is True:
            tot_ke+=0.5*self._container.mass()*(np.dot(self._container.vel(),self._container.vel()))

        return tot_ke

    def momentum(self,include_container=True):
        """Computes the total vector momenta of particles (and container).

        Args:
            include_container (bool): Option to include the momentum of the container

        Returns:
            tuple[float, float]: (x,y) vector component of momentum
        """
        p_x = 0.
        p_y = 0.
        for b in self._balls:
            p_x += b.mass()*b.vel()[0]
            p_y += b.mass()*b.vel()[1]
        if include_container is True:
            p_x += self._container.mass()*self._container.vel()[0]
            p_y += self._container.mass()*self._container.vel()[1]
            return (p_x,p_y)
        return (p_x,p_y)

    def time(self):
        """Get current simulation time

        Returns:
            float: Simulation time"""
        return self._time

    def pressure(self):
        """Computes the current pressure exerted on the container.

        Calls the dp_tot() method on the container object to compute
        total running impuse exerted on the container from the balls
        to calculate the instantaneous pressure.

        Returns:
            float: Pressure
        """
        if self._time == 0:
            return 0.
        pressure = self.container().dp_tot()/(self.container().surface_area()*self.time())
        return pressure

    def t_equipartition(self):
        """Compute the temperature of the system from equipartition theory

        Return:
            float: Equipartition temperature
        """
        ke_average = self.kinetic_energy(include_container=True)/len(self.balls())
        t_theory = ke_average/K_B
        return t_theory

    def t_ideal(self):
        """Compute the temperature of the system based on Ideal Gas Law

        Return:
            float: Ideal Gas Law temperature
        """
        t = self.pressure() * self.container().volume()/(K_B*len(self.balls()))
        return t

    def speeds(self):
        """Get the list of speeds of all the balls

        Return:
            list[float]: speeds of each ball
        """
        ball_speeds = []
        for ball in self._balls:
            speed = np.sqrt(ball.vel()[0]**2 + ball.vel()[1]**2)
            ball_speeds.append(speed)
        return ball_speeds


class BrownianSimulation(MultiBallSimulation):
    """A multiballsimulation of a big ball with many small balls in a container

    Inherited from the MultiBallSimulation class with a bigger ball replacing
    a smaller ball. The time and positions of the big ball are tracked and updated
    with each collision
    """

    def __init__(self,
    c_radius=15,bb_radius=2,bb_mass=10,bb_at_centre=True,rmax=13):
        """Initialise a MultiBallSimulation with one Brownian ball

        Args:
            c_radius (float): Radius of the container
            bb_radius (float): Radius of the brownian ball
            bb_mass (float): Mass of the brownian ball
            bb_at_centre (bool): If the brownian ball starts in the centre
            rmax (int): Radius of the outermost ring for initialisation
        """
        MultiBallSimulation.__init__(self,c_radius,rmax=rmax)
        if bb_at_centre is True:
            ball_replaced = self._balls.pop(0)
        else:
            ball_index = np.random.default_rng().integers(len(self._balls))
            ball_replaced = self._balls.pop(ball_index)
        bb_pos = ball_replaced.pos().copy()
        bb_vel = ball_replaced.pos().copy()

        bb = Ball(pos=bb_pos,vel=bb_vel,radius=bb_radius,mass=bb_mass,colour="Green")
        self._bb = bb
        self._balls.append(self._bb)

        self._bb_positions = [bb_pos]
        self._bb_times = [0.0]

    def bb_positions(self):
        """Get the positions of the big ball

        Return:
            tuple[float, float]: (x,y) position vector
        """
        return self._bb_positions

    def bb_times(self):
        """Get the times of the big ball collisions

        Return:
            list[float]: times of collisions
        """
        return self._bb_times

    def next_collision(self):
        """Perform a collision and update big ball data

        Return:
            float: time of next collision"""
        t =  MultiBallSimulation.next_collision(self)
        self._bb_times.append(self._time)
        self._bb_positions.append(self._bb.pos().copy())
        return t
