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

        # compute number of time steps and size of time steps
        self.compute_times(pars)
        
        # extract number of time steps if happy with the value in
        # pars or trim the time array
        if Nt == None:
            Nt = pars.computational["Nt"]
        else:
            self.t = self.t[:Nt]

        # compute the spatial grid points
        self.r = (1 + np.flip(np.cos(np.linspace(0, np.pi, N)))) / 2

        # Preallocate NumPy arrays for solution components
        self.u = np.zeros((N, Nt + 1))
        self.p = np.zeros((N, Nt + 1))
        self.lam_z = np.ones(Nt + 1)
        self.F = np.zeros(Nt + 1)
        self.J = np.ones((N, Nt + 1))
        self.phi = pars.physical["phi_0"] * np.ones((N, Nt + 1))
        self.fluid_load_fraction = np.zeros(Nt + 1)

    def __str__(self):
        output = (
                    'Solution object with attributes\n'
                  f't: time\n'
                  f'r: radial coordinate in the undeformed state\n'
                  f'u: radial displacement\n'
                  f'p: fluid pressure\n'
                  f'lam_z: axial stretch\n'
                  f'F: force on the platten\n'
                  f'J: Jacobian determinant\n'
                  f'phi: porosity\n'
                  f'fluid_load_fraction: fluid load fraction'
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
            self.t = np.r_[0, np.logspace(-4, np.log10(pars.physical["t_end"]), pars.computational["Nt"])]
        elif pars.computational["t_scaling"] == 'lin':
            self.t = np.linspace(0, pars.physical["t_end"], pars.computational["Nt"] + 1)
        
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