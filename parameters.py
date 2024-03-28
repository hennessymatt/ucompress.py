import numpy as np


class StandardParameters():

    def __init__(self):

        """
        physical parameters
        """

        self.phi_0 = 0.8
        self.lam_z = 0.5
        self.F = -10

        """
        computational parameters
        """
        # parameters for spatial discretisation
        self.N = 40

        # parameters for time stepping
        self.Nt = 100
        self.t = np.r_[0, np.logspace(-4, 1, self.Nt)]
        # self.t = np.linspace(0, 1, self.Nt+1)
        self.dt = np.diff(self.t)

        """
        parameters for Newton's method
        """
        self.newton_max_iterations = 10
        self.newton_conv_tol = 1e-6
