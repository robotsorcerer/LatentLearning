__all__ = ["DoubleIntegrator"]

__author__ = "Lekan Molux"
__date__ = "Nov. 22, 2021"

import numpy as np

class DoubleIntegrator():
    def __init__(self, grid, u_bound=1):
        """
            The base function for the double integrator's 
            minimum time to reach problem.

            Dynamics: \ddot{x}= u,  u \in [-1,1]

            This can represent a car with position 
            x \in \mathbb{R} and with bounded acceleration u acting 
            as the control (negative acceleration corresponds to braking). 
            Let us study the problem of ``parking" the car at the origin, 
            i.e., bringing it to rest at $ x=0$ , in minimal time. 
            It is clear that the system can indeed be brought to rest  
            at the origin from every initial condition. However, since the 
            control is bounded, we cannot do this arbitrarily fast (we are 
            ignoring the trivial case when the system is initialized at the 
            origin). Thus we expect that there exists an optimal control 
            u^* which achieves the transfer in the smallest amount of time. 

            Ref: http://liberzon.csl.illinois.edu/teaching/cvoc/node85.html

            Parameters
            ==========
                grid: state space on which we are resolving this integrator dynamics.
        """

        self.grid     = grid
        self.control_law = u_bound
        self.Gamma = self.switching_curve # switching curve

    @property
    def switching_curve(self):
        """
            \Gamma = -(1/2) . x_2 . |x_2|
        """
        self.Gamma = -.5*self.grid.xs[1]*np.abs(self.grid.xs[1])

        return self.Gamma
 
    def hamiltonian(self, t, data, grid_derivs, schemedata): 
        """
            H = \dot{x1} . x2 + \dot{x2} . u + x_0

            Here, x_0 is initial state which is zero.

            Parameters
            ==========
                grid_derivs: Finite difference of grid points computed
                                with upwinding.
        """
        
        return grid_derivs[0]*self.grid.xs[1]+grid_derivs[1]*self.control_law

    def dynamics(self, t, data, derivMin, derivMax, \
                      schemeData, dim):
        """
            Parameters
            ==========
                dim: The dimension of the ode to return.
        """
        x_dot = [
                    self.grid.xs[1],
                    self.control_law * np.zeros_like(self.grid.xs[1])
        ]

        return x_dot[dim]

    def min_time2reach(self):
        """
            Computes the minimum time we need to reach the 
            switching curve:

            x2 + (sqrt(4x_1 + 2 x_2^2).(x_1 > \Gamma)) + 
            (-x_2 + sqrt(2x_2^2 - 4 x_1) . (x_1 < \Gamma) +
            (|x_2| . (x_1 == \Gamma)).
        """

        #be sure to update the switching curve first5tt
        self.switching_curve
        
        #  Compute the current state on or outside of the 
        # switching curve.

        above_curve = self.grid.xs[0]>self.Gamma
        below_curve = self.grid.xs[0]<self.Gamma
        on_curve    = self.grid.xs[0]==self.Gamma

        reach_term1  = (self.grid.xs[1] + np.emath.sqrt(4*self.grid.xs[0] + \
                         2 * self.grid.xs[1]**2))*above_curve
        reach_term2 =  (-self.grid.xs[1]+np.emath.sqrt(-4*self.grid.xs[0] + \
                        2 * self.grid.xs[1]**2) )*below_curve
        reach_term3 = np.abs(self.grid.xs[1]) * on_curve
        
        reach_time = reach_term1.real + reach_term2.real + reach_term3
                      
        return reach_time
           