"""Balls Module
Contains the Ball and Container objects required for 2D elastic collisions simulation
"""
import numpy as np
import matplotlib.patches as pl
class Ball:
    """A ball that can move and collide elastically.

    A ball has a positon, velocity, radius, mass and can compute
    times to collision with other balls or with a container.

    Attributes:
    _pos (NDArray[float, 2x1]): Contains the current (x,y) position vector
    _vel (NDArray[float, 2x1]): Contains the current (x,y) velocity vector
    _radius (float): Radius of the ball
    _mass (float): Mass of the ball
    _patch (Circle): Matplotlib patch of the ball for figure use -
                     can take a colour argument

    """

    def __init__(self,pos=None,vel=None,radius =1, mass=1.,colour="Red"): # pylint: disable=too-many-arguments
        """Initialise a Ball object.

        Args:
            pos (List or NDArray[float, 2x1]): Ball inital (x,y) position vector
            vel (List or NDArray(float, 2x1)): Ball inital (x,y) velocity vector
            radius (float): Ball radius
            mass (float): Ball mass
            colour (str): Ball patch face-colour

        Raises:
            TypeError: If pos or vel are not shape 2x1
        """
        if pos is None:
            pos = [0.0,0.0]
        if vel is None:
            vel = [1.0,0.0]
        pos_array = np.array(pos,dtype=float)
        if pos_array.shape != (2,):
            raise TypeError("Position must take a 2x1 array")
        self._pos = pos_array

        vel_array = np.array(vel,dtype=float)
        if vel_array.shape !=(2,):
            raise TypeError("Velocity must take a 2x1 array")
        self._vel = vel_array

        self._radius = float(radius)
        self._mass = float(mass)
        self._patch = pl.Circle(self._pos,self._radius,fill=True,ec="Black",fc=colour)

    def pos(self):
        """Get the ball's current position.

        Returns:
            numpy.ndarray: A 2x1 array of the (x,y) position vector
        """
        return self._pos

    def radius(self):
        """Get the ball's radius.

        Returns:
            float: Ball radius
        """
        return self._radius

    def mass(self):
        """Get the ball's mass.

        Returns:
            float: Ball mass
        """
        return self._mass

    def vel(self):
        """Get the ball's velocity vector.

        Returns:
            numpy.ndarray: A 2x1 array of the (x,y) velocity vector
        """
        return self._vel

    def set_vel(self,vel):
        """Set the ball's velocity vector.

        Args:
            vel (List or NDArray(float, 2x1)): Ball's new (x,y) velocity vector

        Raises:
            TypeError: If vel is not shape 2x1
        """
        vel_array = np.array(vel, dtype=float)
        if vel_array.shape != (2,):
            raise TypeError("Velocity must take a 2x1 array")
        self._vel = vel_array

    def move(self,delta_t):
        """Advance the ball's position by vel * delta_t.

        Args:
            dt (float): Time step over which to move

        """
        self._pos =  self._pos + self._vel *delta_t
        self._patch.center = self._pos

    def patch(self):
        """Get the ball patch

        Returns:
            matplotlib.patches.Circle: A circle patch
        """
        return self._patch

    def time_to_collision(self,other):
        """Compute the time for collision between the ball and another ball object.

        Calculates the time to collision between ball-ball or ball-container based
        on a quadratic for collision time.

        Args:
            other (Ball or Container): The other object

        Returns:
            float or None:
                Smallest positive root from the quadratic of collision time
                or None if no future collision time
        """
        pos = self.pos() - other.pos()
        vel = self.vel() - other.vel()
        rad = self.radius() + other.radius() #ball on ball
        rad_wall = self.radius() - other.radius() #ball on wall
        if isinstance(self,Container) is True or isinstance(other,Container) is True:
            rad = rad_wall
        a_coeff = np.dot(vel,vel)
        if a_coeff == 0:
            return None
        b_coeff = 2*np.dot(pos,vel)
        c_coeff = np.dot(pos,pos) - rad**2
        disc = b_coeff**2 - 4*a_coeff*c_coeff
        if disc <0:
            return None
        if disc ==0:
            t_0 = float((-b_coeff +np.sqrt(disc))/(2*a_coeff))
            if t_0<=0:
                return None
            return t_0
        t_pos = float((-b_coeff +np.sqrt(disc))/(2*a_coeff))
        t_neg = float((-b_coeff - np.sqrt(disc))/(2*a_coeff))
        t_values = np.array([t_pos,t_neg])
        filtered_t_values = t_values[t_values>1e-9]
        if len(filtered_t_values)==0:
            return None
        return float(np.min(filtered_t_values))

    def collide(self,other):
        """Perform an elastic collision with another Ball object.

        Performs an elastic collision and updates the velocities of
        the objects based on the 2D elastic collision equation

        Args:
            other (Ball or Container): The other object

        """
        rel_v = self.vel() - other.vel()
        rel_pos = self.pos() - other.pos()
        m_sum = self.mass() + other.mass()
        dist_12 = np.dot(rel_pos,rel_pos)
        dist_21 = np.dot(-rel_pos,-rel_pos)
        project_v = np.dot(rel_v,rel_pos)

        #2d collision equation
        new_vel_self = self.vel()-(2*other.mass())/(m_sum)*project_v/(dist_12)*(rel_pos)
        new_vel_other = other.vel()+(2*self.mass())/(m_sum)*project_v/(dist_21)*(rel_pos)
        Ball.set_vel(self,new_vel_self)
        Ball.set_vel(other,new_vel_other)


class Container(Ball):
    """A circular wall that Balls reside in and collide with the inner surface

    Inherited from the Ball class with a large mass.
    Tracks the total momentum transfer from collisions with balls.

    Attributes:
        _radius (float): Radius of the container
        _mass (float): Mass of the container
        _dp (float): Scalar sum of the impulse exerted
        _patch (Circle): Matplotlib patch of the container for figures


    """
    def __init__(self, radius=10, mass=10000000):
        """Initialise a container object centered at the origin.

        Args:
            radius (float): Container radius
            mass (float): Container mass
        """
        Ball.__init__(self,pos=[0.0,0.0],vel=[0.0,0.0],radius=radius,mass=mass)
        self._dp = 0.0
        self._container_patch = pl.Circle(self._pos,self._radius,fill=False,ec="Blue")

    def volume(self):
        """Calculates the 2D volume of the container

        Returns:
            float: π * radius**2
        """
        return np.pi * self._radius**2

    def surface_area(self):
        """Caluates the 2D surface area of the container

        Returns:
            float: 2π*radius
        """
        return 2* np.pi * self._radius

    def patch(self):
        """Get the container patch

        Returns:
            matplotlib.patches.Circle: A circle patch
        """
        return self._container_patch

    def collide(self,other):
        """Computes an elastic collision on the container and accumulates impulse.

        Args:
            other (Ball): The ball colliding with the container
        """
        v_inital = other.vel().copy()
        Ball.collide(self,other)
        delta_p = np.abs(other.mass()*(other.vel()-v_inital))
        # pylint: disable=no-member
        self._dp += np.linalg.norm(delta_p)

    def dp_tot(self):
        """Get the total impululse exerted on the container.

        Returns:
            float: Sum of total impulse
            """
        return self._dp
