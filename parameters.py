import numpy as np


class Parameters():
    """
    A class to store the parameter values for the problem
    """
    def __init__(self):

        """
        Physical parameters.  For displacement-controlled experiments,
        the force does not need to be assigned.  For force-controlled
        experiments, the value of the axial stretch (lam_z) will be
        used as the initial guess when solving the nonlinear problem
        for the instantaneous response
        """
        self.physical = {
            "phi_0": 0.8,   # initial fluid fraction
            "k_0": 1,       # initial permeability
            "lam_z": 0.5,   # axial stretch
            "F": -10,       # force on the platten
            "G_m": 1,       # shear modulus of the matrix
            "G_f": 100,     # shear modulus of the fibres
            "alpha_f": 0.5  # volume fraction of fibres
        }

        """
        computational parameters
        """
        self.computational = {
            "N": 40,            # number of spatial grid points
            "Nt": 100,          # number of time steps
            "t_end": 10,        # end-time of simulation
            "t_spacing": 'log'  # lin or log spacing between time steps
        }

        # compute the time vector using logarithmic (log) or linear (lin)
        # spacing
        if self.computational["t_spacing"] == 'log':
            self.computational["t"] = np.r_[0, np.logspace(-4, np.log10(self.computational["t_end"]), self.computational["Nt"])]
        elif self.computational["t_scaling"] == 'lin':
            self.computational["t"] = np.linspace(0, self.computational["t_end"], self.computational["Nt"] + 1)
        
        # compute the sizes of the time steps
        self.computational["dt"] = np.diff(self.computational["t"])