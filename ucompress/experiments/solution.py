import numpy as np

class Solution():
    """
    Class for storing the outputs of the solvers
    """
    def __init__(self, pars, Nt = None):
        """
        Nt is a default argument to customise the
        length of the array that are created.  This is helpful
        if using the solution structure to store the instantaneous
        or steady-state response, which are effectively valid at a
        single time point.  By default the array length is defined by 
        the variable Nt in parameter dict.
        """

        # extract number of grid points
        N = pars.computational["N"]

        # extract radius of sample
        R = pars.physical["R"]

        # compute number of time steps and size of time steps
        self.compute_times(pars)
        
        # extract number of time steps if happy with the value in
        # pars or trim the time array
        if Nt == None:
            Nt = pars.computational["Nt"]
        else:
            self.t = self.t[:Nt]

        # compute the spatial grid points
        self.r = R * (1 + np.flip(np.cos(np.linspace(0, np.pi, N)))) / 2

        # Preallocate NumPy arrays for solution components
        self.u = np.zeros((N, Nt + 1))
        self.p = np.zeros((N, Nt + 1))
        self.lam_z = np.ones(Nt + 1)
        self.F = np.zeros(Nt + 1)
        self.J = np.ones((N, Nt + 1))
        self.phi = pars.physical["phi_0"] * np.ones((N, Nt + 1))
        self.fluid_load_fraction = np.zeros(Nt + 1)

    def __str__(self):
        np.set_printoptions(threshold = 4, precision = 3)
        output = (
                    'Solution object with attributes\n'
                  f't (time): ' + str(self.t) + '\n'
                  f'r (radial coordinate): '  + str(self.r) + '\n'
                  f'u (radial displacement): ' + str(self.u) + '\n'
                  f'p (fluid pressure): ' + str(self.p) + '\n'
                  f'lam_z (axial stretch): ' + str(self.lam_z) + '\n'
                  f'F (force on platten): ' + str(self.F) + '\n'
                  f'J (Jacobian determinant): ' + str(self.J) + '\n'
                  f'phi (porosity): ' + str(self.phi) + '\n'
                  f'fluid_load_fraction: ' + str(self.fluid_load_fraction) + '\n'
                  )

        return output

    def compute_times(self, pars):
        """
        Computes a NumPy array of time points that are either
        linearly or logarithmically spaced
        """

        # compute the time vector using logarithmic (log) or linear (lin)
        # spacing
        if pars.computational["t_spacing"] == 'log':
            self.t = np.r_[
                0, 
                np.logspace(
                    np.log10(pars.physical["t_start"]), 
                    np.log10(pars.physical["t_end"]), 
                    pars.computational["Nt"]
                )                            
            ]

        elif pars.computational["t_spacing"] == 'lin':
            self.t = np.linspace(
                pars.physical["t_start"], 
                pars.physical["t_end"], 
                pars.computational["Nt"] + 1
            )
        
        # compute the sizes of the time steps
        self.dt = np.diff(self.t)

    def trim_solution(self, n):
        """
        Trims the solution arrays.  Used when 
        Newton's method doesn't converge
        """
        self.t = self.t[:n]
        self.u = self.u[:, :n]
        self.p = self.p[:, :n]
        self.lam_z = self.lam_z[:n]
        self.F = self.F[:n]
        self.J = self.J[:, :n]
        self.phi = self.phi[:, :n]
        self.fluid_load_fraction = self.fluid_load_fraction[:n]
        
    def redimensionalise(self, pars):
        """
        Re-dimensionalises the output using the scaling factors contained
        in the pars object
        """

        self.t *= pars.scaling["time"]
        self.r *= pars.scaling["space"]
        self.u *= pars.scaling["space"]
        self.p *= pars.scaling["stress"]
        self.F *= pars.scaling["force"]